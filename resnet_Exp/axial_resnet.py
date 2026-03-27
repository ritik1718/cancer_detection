
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import matplotlib.pyplot as plt
import numpy as np
import math
import copy

# Add parent dir to path for src imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.dataset import OralCancerDataset, get_transforms

# Speed optimizations for RTX 40-series
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

# ============================================
# AXIAL + RESNET HYBRID: Axial Attention + ResNet50 Fusion
# Description: Concatenation fusion of full ResNet50 features (local)
#              and Axial Attention features (global row/column context).
# Architecture: ResNet50 (2048) + Axial Branch (256) → Concat (2304) → Classifier
# Optimizer: AdamW + CosineAnnealing with Linear Warmup + AMP + GradClip
# ============================================

IMG_SIZE = 256
BATCH_SIZE = 8
ACCUM_STEPS = 4        # Effective batch size = 8 * 4 = 32
NUM_EPOCHS = 50
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

# Warmup config
WARMUP_EPOCHS = 5
COSINE_ETA_MIN = 1e-6

# Training optimizations
GRAD_CLIP_MAX_NORM = 1.0
LABEL_SMOOTHING = 0.1

# Paths
DATA_DIR = r'c:\Projects\oral-cancer-detection\data\raw'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
TEST_DIR = os.path.join(DATA_DIR, 'test')

MODEL_SAVE_DIR = r'c:\Projects\oral-cancer-detection\resnet_Exp'
EXPERIMENT_NAME = 'axial_resnet'
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, f'{EXPERIMENT_NAME}_best.pth')

CLASS_NAMES = ['Normal', 'OSCC']
NUM_CLASSES = 2


# ============================================
# Axial Attention Components
# ============================================

class AxialAttention(nn.Module):
    """Single-axis self-attention (applied along height OR width)."""

    def __init__(self, in_channels, num_heads=8, axis='height'):
        super(AxialAttention, self).__init__()
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.head_dim = in_channels // num_heads
        self.axis = axis

        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"

        self.qkv = nn.Conv1d(in_channels, in_channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm1d(in_channels)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        if self.axis == 'height':
            x_perm = x.permute(0, 3, 1, 2).reshape(B * W, C, H)
            seq_len = H
            spatial = W
        else:
            x_perm = x.permute(0, 2, 1, 3).reshape(B * H, C, W)
            seq_len = W
            spatial = H

        qkv = self.qkv(x_perm)
        qkv = qkv.reshape(-1, 3, self.num_heads, self.head_dim, seq_len)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        q = q.permute(0, 1, 3, 2)
        attn = torch.matmul(q, k) / self.scale
        attn = torch.softmax(attn, dim=-1)

        v = v.permute(0, 1, 3, 2)
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2)
        out = out.reshape(-1, C, seq_len)

        out = self.proj(out)
        out = self.norm(out)

        if self.axis == 'height':
            out = out.reshape(B, W, C, H).permute(0, 2, 3, 1)
        else:
            out = out.reshape(B, H, C, W).permute(0, 2, 1, 3)

        return out


class AxialAttentionBlock(nn.Module):
    """Full axial attention block: Height-Attention → Width-Attention → FFN."""

    def __init__(self, in_channels, num_heads=8):
        super(AxialAttentionBlock, self).__init__()
        self.height_attn = AxialAttention(in_channels, num_heads=num_heads, axis='height')
        self.width_attn = AxialAttention(in_channels, num_heads=num_heads, axis='width')
        self.ffn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 4, kernel_size=1),
            nn.BatchNorm2d(in_channels * 4),
            nn.GELU(),
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
        )
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.norm2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x = x + self.norm1(self.height_attn(x))
        x = x + self.norm2(self.width_attn(x))
        x = x + self.ffn(x)
        return x


class AxialResNetHybrid(nn.Module):
    """
    Hybrid model fusing full ResNet50 features (local) with
    Axial Attention branch features (global row/column context).
    Architecture: ResNet50 (2048) + Axial (256) → Concat (2304) → Classifier
    """

    def __init__(self, num_classes=2, pretrained=True, num_axial_blocks=4, axial_channels=256, num_heads=8):
        super(AxialResNetHybrid, self).__init__()

        # ---- Load ResNet50 ONCE and share layers ----
        if pretrained:
            resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet50(weights=None)

        # ---- Branch 1: Full ResNet50 (local features) ----
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-1])  # (B, 2048, 1, 1)
        resnet_dim = 2048

        # ---- Branch 2: Axial Attention (global context) ----
        self.axial_stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        # Copy pretrained weights from the shared ResNet
        self.axial_stem[0].weight.data.copy_(resnet.conv1.weight.data)
        self.axial_stem[1].weight.data.copy_(resnet.bn1.weight.data)
        self.axial_stem[1].bias.data.copy_(resnet.bn1.bias.data)
        self.axial_stem[1].running_mean.copy_(resnet.bn1.running_mean)
        self.axial_stem[1].running_var.copy_(resnet.bn1.running_var)

        self.axial_layer1 = copy.deepcopy(resnet.layer1)  # Output: (B, 256, H/4, W/4)

        stem_out_channels = 256
        if axial_channels != stem_out_channels:
            self.channel_proj = nn.Sequential(
                nn.Conv2d(stem_out_channels, axial_channels, kernel_size=1),
                nn.BatchNorm2d(axial_channels),
                nn.ReLU(),
            )
        else:
            self.channel_proj = nn.Identity()

        self.axial_blocks = nn.Sequential(
            *[AxialAttentionBlock(axial_channels, num_heads=num_heads) for _ in range(num_axial_blocks)]
        )
        self.axial_pool = nn.AdaptiveAvgPool2d((1, 1))
        axial_dim = axial_channels

        # ---- Fusion: Concatenation ----
        self.fusion_dim = resnet_dim + axial_dim  # 2048 + 256 = 2304

        # ---- Enhanced Classifier (deeper + stronger regularization) ----
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Branch 1: Full ResNet50
        resnet_feat = self.resnet_features(x)
        resnet_feat = torch.flatten(resnet_feat, 1)  # (B, 2048)

        # Branch 2: Axial Attention
        axial_feat = self.axial_stem(x)
        axial_feat = self.axial_layer1(axial_feat)  # (B, 256, H/4, W/4)
        axial_feat = self.channel_proj(axial_feat)
        axial_feat = self.axial_blocks(axial_feat)
        axial_feat = self.axial_pool(axial_feat)
        axial_feat = torch.flatten(axial_feat, 1)  # (B, 256)

        # Concatenation fusion
        combined = torch.cat((resnet_feat, axial_feat), dim=1)  # (B, 2304)

        return self.classifier(combined)


# ============================================
# Dynamic LR Scheduler: Linear Warmup + Cosine Annealing
# ============================================

class WarmupCosineScheduler(optim.lr_scheduler._LRScheduler):
    """Linear warmup for warmup_steps, then cosine decay to eta_min."""

    def __init__(self, optimizer, warmup_steps, total_steps, eta_min=1e-6, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            scale = (self.last_epoch + 1) / self.warmup_steps
            return [base_lr * scale for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [self.eta_min + (base_lr - self.eta_min) * cosine_decay for base_lr in self.base_lrs]


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, loss, accuracy, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
        'experiment': EXPERIMENT_NAME
    }
    torch.save(checkpoint, filepath)
    print(f'✅ Checkpoint saved to {filepath}')


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, scaler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad(set_to_none=True)

    progress_bar = tqdm(dataloader, desc='Training')
    for step, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        # Mixed precision forward pass
        with autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels) / ACCUM_STEPS  # Scale loss by accum steps

        # Scaled backward pass (accumulate gradients)
        scaler.scale(loss).backward()

        # Step optimizer every ACCUM_STEPS
        if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(dataloader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        running_loss += loss.item() * ACCUM_STEPS
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        current_lr = optimizer.param_groups[0]['lr']
        progress_bar.set_postfix({
            'loss': f'{running_loss/len(dataloader):.4f}',
            'acc': f'{100.*correct/total:.2f}%',
            'lr': f'{current_lr:.2e}'
        })

    return running_loss / len(dataloader), 100. * correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Validation')
        for images, labels in progress_bar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress_bar.set_postfix({'loss': f'{running_loss/len(dataloader):.4f}', 'acc': f'{100.*correct/total:.2f}%'})

    return running_loss / len(dataloader), 100. * correct / total


def main():
    print(f"🔬 EXPERIMENT START: {EXPERIMENT_NAME}")
    print(f"   Description: Axial Attention + ResNet50 Hybrid (Optimized)")
    print(f"   Config: BS={BATCH_SIZE}, Epochs={NUM_EPOCHS}, LR={LEARNING_RATE}")
    print(f"   Optimizations: AMP, GradClip, CosineAnnealing+Warmup, LabelSmoothing, DifferentialLR")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")

    # Dataset
    train_dataset = OralCancerDataset(TRAIN_DIR, transform=get_transforms(train=True, img_size=IMG_SIZE), extra_normal_aug=True)
    val_dataset = OralCancerDataset(VAL_DIR, transform=get_transforms(train=False, img_size=IMG_SIZE), extra_normal_aug=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    # Model
    model = AxialResNetHybrid(num_classes=NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"   Parameters: {total_params:.1f}M")

    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # Differential learning rates: lower LR for pretrained backbone, higher for head
    backbone_params = list(model.resnet_features.parameters()) + list(model.axial_stem.parameters()) + list(model.axial_layer1.parameters())
    head_params = list(model.channel_proj.parameters()) + list(model.axial_blocks.parameters()) + list(model.classifier.parameters())

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': LEARNING_RATE * 0.1},   # Lower LR for pretrained layers
        {'params': head_params, 'lr': LEARNING_RATE},              # Full LR for new layers
    ], weight_decay=WEIGHT_DECAY)

    # Dynamic LR: Linear warmup + Cosine annealing (per-iteration stepping)
    steps_per_epoch = len(train_loader)
    total_steps = NUM_EPOCHS * steps_per_epoch
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        eta_min=COSINE_ETA_MIN
    )

    # Mixed precision scaler
    scaler = GradScaler('cuda')

    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        current_lr = optimizer.param_groups[0]['lr']
        head_lr = optimizer.param_groups[1]['lr']
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} (Backbone LR: {current_lr:.2e}, Head LR: {head_lr:.2e})")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"  Summary: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  ⭐ New Best Model! Acc: {val_acc:.2f}%")
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, val_loss, val_acc, BEST_MODEL_PATH)

    print(f"\n{'='*60}")
    print(f"Experiment {EXPERIMENT_NAME} Complete. Best Acc: {best_val_acc:.2f}%")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
