import os
import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import albumentations as A
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import models
from torch.optim.lr_scheduler import CosineAnnealingLR
import multiprocessing
from torchsummary import summary

# Set random seed
torch.manual_seed(55)
np.random.seed(55)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Global parameters
H = 128
W = 128
BATCH_SIZE = 8  # 批量大小
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
CLASSES = 1
ACCUMULATION_STEPS = 2
NUM_WORKERS = 0  # 设置为0，禁用多线程数据加载


# Create directory
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# Dataset class
class WaterBodyDataset(Dataset):
    def __init__(self, image_paths, mask_paths, augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.augment = augment
        if self.augment:
            self.transform = A.Compose([
                A.Rotate(limit=45, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.3),
                A.GaussNoise(p=0.2),
                A.CLAHE(p=0.3),
                A.RandomResizedCrop(size=(H,W),scale=(0.08, 1.0), ratio=(0.75, 1.333), p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5),
            ],is_check_shapes=False)
        else:
            self.transform = A.Compose([
                A.Resize(height=H, width=W)
            ],is_check_shapes=False)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        img = cv2.resize(img, (W, H))  # 确保图像尺寸为128x128
        img = img.astype(np.float32) / 255.0

        mask_path = self.mask_paths[idx]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")
        mask = cv2.resize(mask, (W, H))  # 确保掩码尺寸也为128x128
        mask = (mask > 127.5).astype(np.float32)
        # 移除这行: mask = np.expand_dims(mask, axis=0)  # 不再需要额外增加维度

        # 确保图像和掩码尺寸一致
        assert img.shape[:2] == mask.shape, f"Image and mask shapes don't match: {img.shape[:2]} vs {mask.shape}"

        if self.augment:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']
        else:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']

        img = torch.from_numpy(img).permute(2, 0, 1).float()  # 添加permute将HWC转为CHW
        mask = torch.from_numpy(mask).float().unsqueeze(0)  # 在这里增加通道维度
        return img, mask


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




# Metrics (保持不变)
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


# Combined Loss (保持不变)
class CombinedLoss(nn.Module):
    def __init__(self, w_bce=1.0, w_dice=1.0, w_focal=1.0, alpha=0.25, gamma=2.0):
        super(CombinedLoss, self).__init__()
        self.w_bce = w_bce
        self.w_dice = w_dice
        self.w_focal = w_focal
        self.bce = nn.BCELoss()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)

    def forward(self, y_pred, y_true):
        # 确保y_true的形状与y_pred一致
        if y_true.size() != y_pred.size():
            y_true = y_true.unsqueeze(1)  # 增加通道维度

        # BCE Loss
        bce_loss = self.bce(y_pred, y_true) if self.w_bce > 0 else 0

        # Dice Loss
        dice_loss = (1 - dice_coef(y_true, y_pred)) if self.w_dice > 0 else 0

        # Focal Loss
        focal_loss = self.focal(y_pred, y_true) if self.w_focal > 0 else 0

        # 加权组合
        total_loss = self.w_bce * bce_loss + self.w_dice * dice_loss + self.w_focal * focal_loss
        return total_loss


# Training function (保持不变)
def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, device):
    best_iou = 0.0
    history = {'train_loss': [], 'val_loss': [],
               'train_iou': [], 'val_iou': [],
               'train_precision': [], 'val_precision': [],
               'train_accuracy': [], 'val_accuracy': [],
               'learning_rate': []}

    create_dir("files")
    if os.path.exists("files/training_log.txt"):
        os.remove("files/training_log.txt")

    # 添加余弦退火调度器
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        train_precision = 0.0
        train_accuracy = 0.0
        for i, (images, masks) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, masks)
            loss = loss / ACCUMULATION_STEPS
            loss.backward()

            if (i + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item() * images.size(0) * ACCUMULATION_STEPS
            train_iou += iou(masks, outputs).item() * images.size(0)
            train_precision += precision(masks, outputs).item() * images.size(0)
            train_accuracy += accuracy(masks, outputs).item() * images.size(0)

        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rate'].append(current_lr)


        train_loss /= len(train_loader.dataset)
        train_iou /= len(train_loader.dataset)
        train_precision /= len(train_loader.dataset)
        train_accuracy /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_precision = 0.0
        val_accuracy = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
                val_iou += iou(masks, outputs).item() * images.size(0)
                val_precision += precision(masks, outputs).item() * images.size(0)
                val_accuracy += accuracy(masks, outputs).item() * images.size(0)

        val_loss /= len(val_loader.dataset)
        val_iou /= len(val_loader.dataset)
        val_precision /= len(val_loader.dataset)
        val_accuracy /= len(val_loader.dataset)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)
        history['train_precision'].append(train_precision)
        history['val_precision'].append(val_precision)
        history['train_accuracy'].append(train_accuracy)
        history['val_accuracy'].append(val_accuracy)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, LR: {current_lr:.2e}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Train mIoU: {train_iou:.4f}, Val mIoU: {val_iou:.4f}, "
            f"Train Precision: {train_precision:.4f}, Val Precision: {val_precision:.4f}, "
            f"Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")

        with open("files/training_log.txt", "a") as f:
            f.write(f"Epoch {epoch + 1}/{num_epochs}, LR: {current_lr:.2e}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                    f"Train mIoU: {train_iou:.4f}, Val mIoU: {val_iou:.4f}, "
                    f"Train Precision: {train_precision:.4f}, Val Precision: {val_precision:.4f}, "
                    f"Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}\n")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), "files/best_model.pth")

        if epoch > 5 and val_iou < history['val_iou'][epoch - 1]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
                print(f"Reduced learning rate to {param_group['lr']}")

    return history

def detailed_params(model):
    print("\n" + "="*50 + " 逐层参数量明细 " + "="*50)
    total = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name:60s}: {param.numel():,}")
            total += param.numel()
    print(f"\n可训练参数总量: {total:,}")


# Main (保持不变)
if __name__ == "__main__":
    # Dataset paths
    base_path = r"C:\Users\Administrator\Desktop\bishe\Water_body_segmentation-DeepLabV3plus\dataset2"
    train_images_path = os.path.normpath(os.path.join(base_path, "train", "images"))
    val_images_path = os.path.normpath(os.path.join(base_path, "val", "images"))
    train_masks_path = os.path.normpath(os.path.join(base_path, "train", "masks"))
    val_masks_path = os.path.normpath(os.path.join(base_path, "val", "masks"))

    # Verify paths
    paths = {
        "train_images": train_images_path,
        "train_masks": train_masks_path,
        "val_images": val_images_path,
        "val_masks": val_masks_path
    }
    for name, path in paths.items():
        if not os.path.exists(path):
            print(f"ERROR: Path does not exist: {path}")
            print(
                "Please ensure the dataset is placed in the correct directory with subfolders: train/images, train/masks, val/images, val/masks")
            print("Expected structure:")
            print(f"{base_path}\\train\\images\\*.jpg")
            print(f"{base_path}\\train\\masks\\*.jpg")
            print(f"{base_path}\\val\\images\\*.jpg")
            print(f"{base_path}\\val\\masks\\*.jpg")
            exit(1)
        else:
            files = glob(os.path.join(path, "*.jpg"))
            print(f"{name}: {len(files)} files")
            if files:
                print(f"Sample files: {files[:2]}")
            else:
                print(f"WARNING: No .jpg files found in {path}")

    # Load data
    train_image_paths = sorted(glob(os.path.join(train_images_path, "*.jpg")))
    train_mask_paths = sorted(glob(os.path.join(train_masks_path, "*.jpg")))
    val_image_paths = sorted(glob(os.path.join(val_images_path, "*.jpg")))
    val_mask_paths = sorted(glob(os.path.join(val_masks_path, "*.jpg")))

    if not all([train_image_paths, train_mask_paths, val_image_paths, val_mask_paths]):
        print("ERROR: One or more dataset paths contain no .jpg files.")
        print("Please verify the dataset structure and ensure .jpg files exist in all subfolders.")
        exit(1)

    # Verify image-mask pairing
    train_image_names = [os.path.basename(p) for p in train_image_paths]
    train_mask_names = [os.path.basename(p) for p in train_mask_paths]
    val_image_names = [os.path.basename(p) for p in val_image_paths]
    val_mask_names = [os.path.basename(p) for p in val_mask_paths]

    if set(train_image_names) != set(train_mask_names):
        print("WARNING: Train images and masks do not match.")
        print("Please ensure each image has a corresponding mask with the same filename.")
    if set(val_image_names) != set(val_mask_names):
        print("WARNING: Validation images and masks do not match.")
        print("Please ensure each image has a corresponding mask with the same filename.")

    train_image_paths, train_mask_paths = shuffle(train_image_paths, train_mask_paths, random_state=42)

    print(f"Train images: {len(train_image_paths)}, Val images: {len(val_image_paths)}")
    print(f"Train masks: {len(train_mask_paths)}, Val masks: {len(val_mask_paths)}")

    # Create datasets
    train_dataset = WaterBodyDataset(train_image_paths, train_mask_paths, augment=True)
    val_dataset = WaterBodyDataset(val_image_paths, val_mask_paths, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Model
    model = DeepLabV3Plus(classes=CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = CombinedLoss(w_bce=1.0, w_dice=1.0, w_focal=0.8, alpha=0.7, gamma=2.0)

    detailed_params(model)

    # Train
    history = train_model(model, train_loader, val_loader, optimizer, criterion, NUM_EPOCHS, device)

    # Plot results
    plt.figure(figsize=(15, 12))

    plt.subplot(3, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(history['train_iou'], label='Training mIoU')
    plt.plot(history['val_iou'], label='Validation mIoU')
    plt.title('Training and Validation mIoU')
    plt.xlabel('Epochs')
    plt.ylabel('mIoU')
    plt.legend()

    plt.subplot(3, 2, 3)
    plt.plot(history['train_precision'], label='Training Precision')
    plt.plot(history['val_precision'], label='Validation Precision')
    plt.title('Training and Validation Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(history['train_accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # 添加学习率曲线
    plt.subplot(3, 2, 5)
    plt.plot(history['learning_rate'], label='Learning Rate', color='purple')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.legend()

    plt.tight_layout()
    plt.savefig('files/metrics_plot.png')
    plt.clf()