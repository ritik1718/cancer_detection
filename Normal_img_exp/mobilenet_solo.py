
import os
import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights
import numpy as np

# Add parent dir to path for src imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.dataset import OralCancerDataset, get_transforms

# Speed optimizations for RTX 40-series
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

# ============================================
# MOBILENET SOLO: Standalone MobileNetV2
# Description: Fine-tuned MobileNetV2 (ImageNet pretrained) for oral cancer detection.
#              Lightweight and efficient architecture.
# Architecture: MobileNetV2 → Global Avg Pool → Enhanced Classifier
# Feature dim: 1280
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
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_SCRIPT_DIR, 'data', 'raw')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
TEST_DIR = os.path.join(DATA_DIR, 'test')

MODEL_SAVE_DIR = _SCRIPT_DIR
EXPERIMENT_NAME = 'mobilenet_solo'
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, f'{EXPERIMENT_NAME}_best.pth')

CLASS_NAMES = ['Normal', 'OSCC']
NUM_CLASSES = 2


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
            scale = (self.last_epoch + 1) / self.warmup_steps
            return [base_lr * scale for base_lr in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [self.eta_min + (base_lr - self.eta_min) * cosine_decay for base_lr in self.base_lrs]


class MobileNetSolo(nn.Module):
    """Standalone MobileNetV2 fine-tuned for binary oral cancer classification."""

    def __init__(self, num_classes=2, pretrained=True):
        super(MobileNetSolo, self).__init__()

        # MobileNetV2 backbone
        if pretrained:
            self.mobilenet = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        else:
            self.mobilenet = models.mobilenet_v2(weights=None)

        # MobileNetV2 features output: 1280 channels
        mobilenet_dim = 1280

        # Replace the default classifier with enhanced head
        self.mobilenet.classifier = nn.Sequential(
            nn.Linear(mobilenet_dim, 512),
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
        return self.mobilenet(x)


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

        with autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels) / ACCUM_STEPS

        scaler.scale(loss).backward()

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
    print(f"   Description: Standalone MobileNetV2 Fine-tuned (Optimized)")
    print(f"   Config: BS={BATCH_SIZE}x{ACCUM_STEPS}={BATCH_SIZE*ACCUM_STEPS}, Epochs={NUM_EPOCHS}, LR={LEARNING_RATE}")
    print(f"   Optimizations: AMP, GradClip, CosineAnnealing+Warmup, LabelSmoothing, DifferentialLR")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")

    # Dataset
    train_dataset = OralCancerDataset(TRAIN_DIR, transform=get_transforms(train=True, img_size=IMG_SIZE), extra_normal_aug=True)
    val_dataset = OralCancerDataset(VAL_DIR, transform=get_transforms(train=False, img_size=IMG_SIZE), extra_normal_aug=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    # Model
    model = MobileNetSolo(num_classes=NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"   Parameters: {total_params:.1f}M")

    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # Differential learning rates
    backbone_params = list(model.mobilenet.features.parameters())
    head_params = list(model.mobilenet.classifier.parameters())

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': LEARNING_RATE * 0.1},
        {'params': head_params, 'lr': LEARNING_RATE},
    ], weight_decay=WEIGHT_DECAY)

    # Dynamic LR: Linear warmup + Cosine annealing
    steps_per_epoch = len(train_loader)
    total_steps = NUM_EPOCHS * steps_per_epoch
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch

    scheduler = WarmupCosineScheduler(
        optimizer, warmup_steps=warmup_steps, total_steps=total_steps, eta_min=COSINE_ETA_MIN
    )

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
