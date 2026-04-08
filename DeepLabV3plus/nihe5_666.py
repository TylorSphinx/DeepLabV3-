import os
import numpy as np
import cv2
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models  # 确保导入 transforms 和 models
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# 确保测试脚本使用相同的设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 根据实际情况调整

H = 128
W = 128

# ECA注意力机制实现
class ECA(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        kernel_size = int(abs((gamma * torch.log(torch.tensor(channels, dtype=torch.float)) + b) / 2))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


# 混合池化层
class HybridPool(nn.Module):
    def __init__(self, in_channels, pool_size=7, reduction=16):
        super().__init__()
        self.pool_size = pool_size
        self.avg_pool = nn.AdaptiveAvgPool2d(pool_size)
        self.max_pool = nn.AdaptiveMaxPool2d(pool_size)

        # 确保reduction后的通道数至少为1
        mid_channels = max(1, in_channels // reduction)

        # 动态权重生成
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, 2, 1),
            nn.Sigmoid()
        )

        # 通道压缩 - 修改为处理3倍输入通道数
        self.compress = nn.Conv2d(in_channels * 3, in_channels, 1)

    def forward(self, x):
        # 获取输入特征图的高度和宽度
        _, _, h, w = x.shape

        # 自适应池化
        avg_out = self.avg_pool(x)  # [B, C, pool_size, pool_size]
        max_out = self.max_pool(x)  # [B, C, pool_size, pool_size]

        # 生成注意力权重并调整形状以匹配池化输出
        weights = self.attention(x)  # [B, 2, 1, 1]
        weights = F.interpolate(weights, size=(self.pool_size, self.pool_size),
                                mode='bilinear', align_corners=False)  # [B, 2, pool_size, pool_size]

        # 动态融合
        fused = weights[:, 0:1] * avg_out + weights[:, 1:2] * max_out  # [B, C, pool_size, pool_size]

        # 多尺度增强 - 创建3个不同尺度的特征图
        multi_scale = [
            F.interpolate(fused, size=(self.pool_size, self.pool_size), mode='bilinear', align_corners=False),
            F.interpolate(fused, size=(self.pool_size * 2, self.pool_size * 2), mode='bilinear', align_corners=False),
            F.interpolate(fused, size=(self.pool_size * 4, self.pool_size * 4), mode='bilinear', align_corners=False)
        ]

        # 将所有特征图上采样到最大尺寸
        max_size = (self.pool_size * 4, self.pool_size * 4)
        multi_scale = [F.interpolate(feat, size=max_size, mode='bilinear', align_corners=False)
                       for feat in multi_scale]

        # 拼接并压缩 - 现在输入通道数是in_channels*3
        return self.compress(torch.cat(multi_scale, dim=1))


# 修改后的多尺度特征融合模块
class MultiScaleFusion(nn.Module):
    def __init__(self, in_channels, scales=[1, 2, 4], reduced_channels=256):
        super().__init__()
        self.scales = scales
        self.reduced_channels = reduced_channels

        # 先用1x1卷积降维（避免后续计算量过大）
        self.channel_reduce = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 1),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True)
        )

        self.branches = nn.ModuleList()
        for s in scales:
            if s == 1:
                self.branches.append(nn.Identity())
            else:
                self.branches.append(nn.Sequential(
                    nn.Upsample(scale_factor=s, mode='bilinear', align_corners=False),
                    # Depthwise Conv
                    nn.Conv2d(reduced_channels, reduced_channels, 3, padding=1, groups=reduced_channels),
                    # Pointwise Conv
                    nn.Conv2d(reduced_channels, reduced_channels, 1),
                    nn.BatchNorm2d(reduced_channels),
                    nn.ReLU(inplace=True)
                ))

        # 融合层（拼接多尺度特征）
        self.fusion = nn.Sequential(
            nn.Conv2d(len(scales) * reduced_channels, reduced_channels, 1),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.channel_reduce(x)  # 先降维 [B, 1152, H, W] -> [B, 256, H, W]
        features = []
        target_size = x.shape[2:]

        for branch in self.branches:
            feat = branch(x)
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            features.append(feat)

        fused = torch.cat(features, dim=1)  # [B, 256 * 3, H, W]
        return self.fusion(fused)  # [B, 256, H, W]


# 修改后的MobileNetV3作为Backbone
class MobileNetV3Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载预训练MobileNetV3 Small
        base_model = models.mobilenet_v3_large(pretrained=False)
        base_model.load_state_dict(
            torch.load(r"C:\Users\Administrator\.cache\torch\hub\checkpoints\mobilenet_v3_large-5c1a4163.pth"))

        # 修改特征提取部分
        self.features = base_model.features
        self.eca = ECA(960)  # MobileNetV3 Small最后一个block的输出通道数
        self.hybrid_pool = HybridPool(in_channels=960, pool_size=7)
        self.ms_fusion = MultiScaleFusion(960)  # 混合池化后通道数翻倍

        # 保存中间特征用于跳跃连接
        self.skip_layers = []
        self._register_hooks()

    def _register_hooks(self):
        def hook(module, input, output, name):
            self.skip_layers.append((name, output))

        # 注册hook获取中间层特征
        self.features[0].register_forward_hook(lambda m, i, o: hook(m, i, o, 'layer0'))
        self.features[4].register_forward_hook(lambda m, i, o: hook(m, i, o, 'layer4'))
        self.features[9].register_forward_hook(lambda m, i, o: hook(m, i, o, 'layer9'))
        self.features[15].register_forward_hook(lambda m, i, o: hook(m, i, o, 'layer15'))

    def forward(self, x):
        input_size = x.shape[2:]
        self.skip_layers = []
        x = self.features(x)
        x = self.eca(x)
        x = self.hybrid_pool(x)
        x = self.ms_fusion(x)  # 多尺度融合
        return x, self._get_skip_features()

    def _get_skip_features(self):
        # 返回最后三个中间层特征
        skip_features = []
        for name, feat in self.skip_layers[-3:]:
            skip_features.append(feat)
        return skip_features


# 修改后的ASPP Module
class ASPP(nn.Module):
    def __init__(self, in_channels = 256, out_channels = 256):
        super(ASPP, self).__init__()
        self.rates = [3, 6, 9]  # 对应实际感受野直径：3, 13, 25

        # 分支0：1x1卷积
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(32, out_channels),  # 替换为GroupNorm
            nn.ReLU(inplace=True)
        )

        # 分支1-3：膨胀卷积
        self.b1 = self._make_atrous_conv(self.rates[0], out_channels)
        self.b2 = self._make_atrous_conv(self.rates[1], out_channels)
        self.b3 = self._make_atrous_conv(self.rates[2], out_channels)

        # 分支4：全局平均池化
        self.b4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(32, out_channels),  # 替换为GroupNorm
            nn.ReLU(inplace=True)
        )

        # 融合层
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def _make_atrous_conv(self, rate, out_channels):
        return nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 获取特征图尺寸（应为16×16）
        h, w = x.shape[2:]

        # 处理各分支
        b0 = self.b0(x)
        b1 = self.b1(x)
        b2 = self.b2(x)
        b3 = self.b3(x)
        b4 = self.b4(x)
        b4 = F.interpolate(b4, size=(h, w), mode='bilinear', align_corners=True)

        # 合并并投影
        x = torch.cat([b0, b1, b2, b3, b4], dim=1)
        return self.project(x)



# Decoder Module (修改为适应多尺度跳跃连接)
class Decoder(nn.Module):
    def __init__(self, skip_channels_list, classes):
        super(Decoder, self).__init__()
        # 修改为每个跳跃连接降维到64通道
        self.skip_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, 64, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=False)
            ) for channels in skip_channels_list
        ])

        # 修改输入通道数为256 (ASPP输出) + 64 * len(skip_channels_list)
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 64 * len(skip_channels_list), 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Conv2d(512, classes, 1),
            nn.Sigmoid()
        )

    def forward(self, x, skip_features, input_size):
        # 上采样并融合所有跳跃连接特征
        upsampled_features = []
        for i, feat in enumerate(skip_features):
            feat = self.skip_convs[i](feat)
            feat = F.interpolate(feat, size=x.shape[2:], mode='bilinear', align_corners=True)
            upsampled_features.append(feat)

        # 拼接所有特征
        fused_features = torch.cat([x] + upsampled_features, dim=1)
        x = self.decoder(fused_features)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        return x


# 修改后的DeepLabV3+ Model
class DeepLabV3Plus(nn.Module):
    def __init__(self, classes=1):
        super(DeepLabV3Plus, self).__init__()
        self.backbone = MobileNetV3Backbone()
        # 修改ASPP的输入通道数为1152，因为MobileNetV3 Small的输出是576*2
        self.aspp = ASPP(in_channels=256, out_channels=256)

        # 获取backbone中间层的通道数
        dummy_input = torch.randn(1, 3, H, W)
        _, skip_features = self.backbone(dummy_input)
        skip_channels = [f.shape[1] for f in skip_features]

        self.decoder = Decoder(skip_channels_list=skip_channels, classes=classes)

    def forward(self, x):
        input_size = x.shape[2:]
        x, skip_features = self.backbone(x)
        x = self.aspp(x)
        x = self.decoder(x, skip_features, input_size)
        return x

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 平衡正负样本的权重（通常为前景类别的逆频率）
        self.gamma = gamma  # 难易样本调节因子（>0时对难样本加权）
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # pt = p if y=1, else 1-p
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

class CombinedLoss(nn.Module):
    def __init__(self, w_bce=1.0, w_dice=1.0, w_focal=1.0, alpha=0.25, gamma=2.0):
        super(CombinedLoss, self).__init__()
        self.w_bce = w_bce
        self.w_dice = w_dice
        self.w_focal = w_focal
        self.bce = nn.BCELoss()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)

    def forward(self, y_pred, y_true):
        # BCE Loss
        bce_loss = self.bce(y_pred, y_true) if self.w_bce > 0 else 0

        # Dice Loss
        dice_loss = (1 - dice_coef(y_true, y_pred)) if self.w_dice > 0 else 0

        # Focal Loss
        focal_loss = self.focal(y_pred, y_true) if self.w_focal > 0 else 0

        # 加权组合
        total_loss = self.w_bce * bce_loss + self.w_dice * dice_loss + self.w_focal * focal_loss
        return total_loss

# 测试数据集类
class WaterBodyDataset(Dataset):
    def __init__(self, image_paths, mask_paths, augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.augment = augment
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        img = cv2.resize(img, (H, W))  # 假设图像大小为100x100
        img = img.astype(np.float32) / 255.0

        mask_path = self.mask_paths[idx]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")
        mask = cv2.resize(mask, (H, W))  # 假设掩码大小为100x100
        mask = (mask > 127.5).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)

        img = self.transform(img).float()
        mask = torch.from_numpy(mask).float()
        return img, mask



# 定义评估指标函数
def kappa_score(y_true, y_pred, smooth=1e-15):
    # 确保输入是PyTorch张量
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)

    y_true = y_true.view(-1)
    y_pred = (y_pred > 0.5).float().view(-1)

    # 构建混淆矩阵
    cm = confusion_matrix(y_true.cpu().numpy(), y_pred.cpu().numpy())
    n = cm.sum()
    sum_po = 0
    sum_pe = 0

    for i in range(len(cm)):
        sum_po += cm[i][i]
        row = np.sum(cm[i, :])
        col = np.sum(cm[:, i])
        sum_pe += row * col

    po = sum_po / n
    pe = sum_pe / (n * n)

    kappa = (po - pe) / (1 - pe + smooth)
    return kappa
def iou(y_true, y_pred, smooth=1e-15):
    y_true = y_true.view(-1)
    y_pred = (y_pred > 0.5).float().view(-1)
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def dice_coef(y_true, y_pred, smooth=1e-15):
    y_true = y_true.view(-1)
    y_pred = (y_pred > 0.5).float().view(-1)
    intersection = (y_true * y_pred).sum()
    return (2. * intersection + smooth) / (y_true.sum() + y_pred.sum() + smooth)


def precision(y_true, y_pred, smooth=1e-15):
    y_true = y_true.view(-1)
    y_pred = (y_pred > 0.5).float().view(-1)
    true_positives = (y_true * y_pred).sum()
    predicted_positives = y_pred.sum()
    return (true_positives + smooth) / (predicted_positives + smooth)


def accuracy(y_true, y_pred, smooth=1e-15):
    y_true = y_true.view(-1)
    y_pred = (y_pred > 0.5).float().view(-1)
    correct_predictions = (y_true == y_pred).sum()
    total_pixels = y_true.shape[0]
    return (correct_predictions + smooth) / (total_pixels + smooth)


def f1_score(y_true, y_pred, smooth=1e-15):
    # 确保输入是 PyTorch 张量
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)

    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    true_positives = (y_true * y_pred).sum()
    predicted_positives = y_pred.sum()
    actual_positives = y_true.sum()

    precision = (true_positives + smooth) / (predicted_positives + smooth)
    recall = (true_positives + smooth) / (actual_positives + smooth)

    return 2 * ((precision * recall) / (precision + recall + smooth))


def recall_score(y_true, y_pred, smooth=1e-15):
    # 确保输入是 PyTorch 张量
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)

    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    true_positives = (y_true * y_pred).sum()
    actual_positives = y_true.sum()
    return (true_positives + smooth) / (actual_positives + smooth)


# 测试函数
def test_model(model, test_loader, device, plot_samples=5):
    model.eval()
    total_metrics = {
        'loss': 0.0,
        'iou': 0.0,
        'dice': 0.0,
        'precision': 0.0,
        'accuracy': 0.0,
        'f1': 0.0,
        'recall': 0.0,
        'specificity': 0.0,
        'kappa': 0.0,
    }

    all_preds = []
    all_targets = []

    sample_images = []
    sample_masks = []
    sample_preds = []

    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(test_loader, desc="Testing")):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            # 计算损失
            loss = criterion(outputs, masks)
            total_metrics['loss'] += loss.item() * images.size(0)

            # 计算各项指标
            preds = (outputs > 0.5).float()
            total_metrics['iou'] += iou(masks, outputs).item() * images.size(0)
            total_metrics['dice'] += dice_coef(masks, outputs).item() * images.size(0)
            total_metrics['precision'] += precision(masks, outputs).item() * images.size(0)
            total_metrics['accuracy'] += accuracy(masks, outputs).item() * images.size(0)
            total_metrics['kappa'] += kappa_score(masks, preds).item() * images.size(0)

            # 确保传递的是张量，而不是 NumPy 数组
            total_metrics['f1'] += f1_score(masks, preds).item() * images.size(0)
            total_metrics['recall'] += recall_score(masks, preds).item() * images.size(0)

            all_preds.extend(preds.view(-1).cpu().numpy())
            all_targets.extend(masks.view(-1).cpu().numpy())

            if i < plot_samples:
                sample_images.append(images.cpu().numpy())
                sample_masks.append(masks.cpu().numpy())
                sample_preds.append(preds.cpu().numpy())

    num_samples = len(test_loader.dataset)
    for metric in total_metrics:
        total_metrics[metric] /= num_samples

    cm = confusion_matrix(all_targets, all_preds)
    tn, fp, fn, tp = cm.ravel()
    total_metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

    print("\nTest Results:")
    print(f"Test Loss: {total_metrics['loss']:.4f}")
    print(f"Mean IoU: {total_metrics['iou']:.4f}")
    print(f"Dice Coefficient: {total_metrics['dice']:.4f}")
    print(f"Precision: {total_metrics['precision']:.4f}")
    print(f"Accuracy: {total_metrics['accuracy']:.4f}")
    print(f"F1 Score: {total_metrics['f1']:.4f}")
    print(f"Recall: {total_metrics['recall']:.4f}")
    print(f"Specificity: {total_metrics['specificity']:.4f}")
    print(f"Kappa Coefficient: {total_metrics['kappa']:.4f}")

    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, digits=4))

    plot_sample_predictions(sample_images, sample_masks, sample_preds)

def plot_sample_predictions(images, masks, preds, num_samples=1):
    plt.figure(figsize=(20, 10))
    for i in range(num_samples):
        plt.subplot(num_samples, 3, i * 3 + 1)
        img = np.transpose(images[i][0], (1, 2, 0))  # 从 (C, H, W) 转换为 (H, W, C)
        img = (img - img.min()) / (img.max() - img.min())  # 归一化
        plt.imshow(img)
        plt.title('Input Image')
        plt.axis('off')

        plt.subplot(num_samples, 3, i * 3 + 2)
        mask = masks[i][0].squeeze()  # 从 (1, H, W) 转换为 (H, W)
        plt.imshow(mask, cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')

        plt.subplot(num_samples, 3, i * 3 + 3)
        pred = preds[i][0].squeeze()  # 从 (1, H, W) 转换为 (H, W)
        plt.imshow(pred, cmap='gray')
        plt.title('Prediction')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('files/sample_predictions.png')
    plt.clf()


class PredictionDataset(Dataset):
    """专门用于预测的数据集类，不需要标签"""

    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        img = cv2.resize(img, (H, W))
        img = img.astype(np.float32) / 255.0
        img = self.transform(img).float()
        return img, os.path.basename(img_path)  # 返回图像和文件名


def predict_images(model, image_paths, output_dir="predictions"):
    """
    对输入图像进行预测并保存结果

    参数:
        model: 训练好的模型
        image_paths: 要预测的图像路径列表
        output_dir: 预测结果保存目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 创建数据集和数据加载器
    dataset = PredictionDataset(image_paths)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    model.eval()
    with torch.no_grad():
        for images, filenames in tqdm(loader, desc="Predicting"):
            images = images.to(device)
            outputs = model(images)

            # 将预测结果转换为二值图像
            preds = (outputs > 0.5).float().cpu().numpy()

            # 保存每张预测结果
            for i in range(preds.shape[0]):
                pred_mask = preds[i][0] * 255  # 从[0,1]转换为[0,255]
                filename = filenames[i]
                save_path = os.path.join(output_dir, f"pred_{filename}")
                cv2.imwrite(save_path, pred_mask)

    print(f"Predictions saved to {output_dir}")


def visualize_predictions(image_paths, output_dir="predictions", num_samples=5):
    """
    可视化预测结果

    参数:
        image_paths: 原始图像路径列表
        output_dir: 预测结果目录
        num_samples: 要可视化的样本数量
    """
    plt.figure(figsize=(15, 5 * num_samples))

    for i in range(min(num_samples, len(image_paths))):
        # 加载原始图像
        img = cv2.imread(image_paths[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (H, W))

        # 加载预测结果
        filename = os.path.basename(image_paths[i])
        pred_path = os.path.join(output_dir, f"pred_{filename}")
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        # 绘制
        plt.subplot(num_samples, 2, 2 * i + 1)
        plt.imshow(img)
        plt.title(f"Input: {filename}")
        plt.axis('off')

        plt.subplot(num_samples, 2, 2 * i + 2)
        plt.imshow(pred, cmap='gray')
        plt.title("Prediction")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "prediction_samples.png"))
    plt.show()


if __name__ == "__main__":
    # 加载模型
    model = DeepLabV3Plus(classes=1).to(device)
    if os.path.exists("files5/best_model.pth"):
        model.load_state_dict(torch.load("files5/best_model.pth", map_location=device))
        print("Loaded trained model successfully.")
    else:
        print("ERROR: No trained model found. Please train the model first.")
        exit(1)

    # 用户输入图像和标签路径
    print("请选择操作模式:")
    print("1. 预测单个图像")
    print("2. 预测整个目录中的图像")
    print("3. 评估模型在测试集上的表现")

    choice = input("请输入选项(1/2/3): ").strip()

    if choice == "1":
        # 预测单个图像
        image_path = input("请输入要预测的图像路径: ").strip()
        if not os.path.exists(image_path):
            print("错误: 指定的图像路径不存在")
            exit(1)

        predict_images(model, [image_path])
        visualize_predictions([image_path])

    elif choice == "2":
        # 预测整个目录
        dir_path = input("请输入包含图像的目录路径: ").strip()
        if not os.path.exists(dir_path):
            print("错误: 指定的目录不存在")
            exit(1)

        image_paths = sorted(glob(os.path.join(dir_path, "*.jpg"))) + \
                      sorted(glob(os.path.join(dir_path, "*.png"))) + \
                      sorted(glob(os.path.join(dir_path, "*.jpeg")))

        if not image_paths:
            print("错误: 目录中没有找到支持的图像文件(.jpg/.png/.jpeg)")
            exit(1)

        predict_images(model, image_paths)
        visualize_predictions(image_paths)

    elif choice == "3":
        # 评估模型 (使用原始测试代码)
        base_path = input("请输入测试数据集根目录路径: ").strip()
        test_images_path = os.path.normpath(os.path.join(base_path, "nihe", "images"))
        test_masks_path = os.path.normpath(os.path.join(base_path, "nihe", "masks"))

        if not os.path.exists(test_images_path) or not os.path.exists(test_masks_path):
            print("ERROR: Test dataset paths do not exist.")
            exit(1)

        test_image_paths = sorted(glob(os.path.join(test_images_path, "*.jpg")))
        test_mask_paths = sorted(glob(os.path.join(test_masks_path, "*.jpg")))

        if not test_image_paths or not test_mask_paths:
            print("ERROR: No test images or masks found.")
            exit(1)

        test_dataset = WaterBodyDataset(test_image_paths, test_mask_paths, augment=False)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

        # 定义损失函数
        criterion = CombinedLoss(w_bce=1.0, w_dice=1.0, w_focal=0.8, alpha=0.7, gamma=2.0)

        # 运行测试
        test_model(model, test_loader, device)

    else:
        print("错误: 无效的选项")