import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm
import torchvision.models as models
from torchvision.models import DenseNet121_Weights
import matplotlib.pyplot as plt
import numpy as np

# Add parent dir to path for src imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.dataset import OralCancerDataset, get_transforms

# ============================================
# HYBRID 3-MODEL CONFIGURATION
# ============================================
IMG_SIZE = 384
BATCH_SIZE = 2          # 3 large backbones — keep VRAM usage manageable
NUM_EPOCHS = 30
LEARNING_RATE = 1e-5    # Low LR for fine-tuning three pretrained models
WEIGHT_DECAY = 1e-4
PATIENCE = 10           # Early stopping patience

# Absolute paths for data directories
DATA_DIR = r'c:\Projects\oral-cancer-detection\data\raw'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# ============================================
# SAVE TO A SEPARATE FOLDER (preserves all previous models)
# ============================================
MODEL_SAVE_DIR = r'c:\Projects\oral-cancer-detection\models\saved_models_hybrid3'
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'best_model_hybrid_3model.pth')

CLASS_NAMES = ['Normal', 'OSCC']
NUM_CLASSES = 2


class Hybrid3Model(nn.Module):
    """
    Hybrid Model combining the best features of THREE architectures:

    1. DenseNet121  — Local feature extraction
       Dense connections preserve fine-grained textures, edges, and local patterns.
       Output: 1024 dims

    2. ViT-Base-384 — Global feature extraction
       Self-attention over full image captures long-range dependencies and context.
       Output: 768 dims (CLS token)

    3. Swin-Base-384 — Hierarchical feature extraction
       Shifted-window attention bridges local and global at multiple scales.
       Output: 1024 dims

    Fusion:
        Concatenate(1024 + 768 + 1024 = 2816) → FC(512) → FC(128) → FC(2)
    """

    def __init__(self, num_classes=2, pretrained=True):
        super(Hybrid3Model, self).__init__()

        print("=" * 60)
        print("Initializing Hybrid 3-Model")
        print("  (DenseNet121 + ViT-Base-384 + Swin-Base-384)")
        print("=" * 60)

        # ---- 1. LOCAL FEATURES: DenseNet121 ----
        # Excellent at capturing textures, edges, micro-patterns
        if pretrained:
            self.densenet = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        else:
            self.densenet = models.densenet121(weights=None)

        self.densenet_features = self.densenet.features
        self.densenet_pooling = nn.AdaptiveAvgPool2d((1, 1))
        densenet_dim = 1024
        print(f"  ✅ DenseNet121 loaded (Local Features: {densenet_dim} dims)")

        # ---- 2. GLOBAL FEATURES: ViT-Base-384 ----
        # Captures long-range dependencies across the full image
        # num_classes=0 → returns CLS token embedding (768 dims)
        self.vit = timm.create_model(
            'vit_base_patch16_384',
            pretrained=pretrained,
            num_classes=0
        )
        vit_dim = 768
        print(f"  ✅ ViT-Base-384 loaded (Global Features: {vit_dim} dims)")

        # ---- 3. HIERARCHICAL FEATURES: Swin-Base-384 ----
        # Shifted-window attention at multiple scales bridges local and global
        # num_classes=0 → returns pooled feature vector (1024 dims)
        self.swin = timm.create_model(
            'swin_base_patch4_window12_384',
            pretrained=pretrained,
            num_classes=0
        )
        swin_dim = 1024
        print(f"  ✅ Swin-Base-384 loaded (Hierarchical Features: {swin_dim} dims)")

        # ---- 4. FUSION CLASSIFIER ----
        self.fusion_dim = densenet_dim + vit_dim + swin_dim  # 1024 + 768 + 1024 = 2816

        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, num_classes)
        )

        total_params = sum(p.numel() for p in self.parameters()) / 1e6
        print(f"  ✅ Fusion Classifier initialized ({self.fusion_dim} → 512 → 128 → {num_classes})")
        print(f"  📊 Total Parameters: {total_params:.1f}M")
        print("=" * 60)

    def forward(self, x):
        # x: (B, 3, 384, 384)

        # --- DenseNet Branch (Local textures & edges) ---
        dn_feat = self.densenet_features(x)        # (B, 1024, H', W')
        dn_feat = self.densenet_pooling(dn_feat)   # (B, 1024, 1, 1)
        dn_feat = torch.flatten(dn_feat, 1)        # (B, 1024)

        # --- ViT Branch (Global context) ---
        vit_feat = self.vit(x)                     # (B, 768)

        # --- Swin Branch (Hierarchical multi-scale) ---
        swin_feat = self.swin(x)                   # (B, 1024)

        # --- Fusion ---
        combined = torch.cat((dn_feat, vit_feat, swin_feat), dim=1)  # (B, 2816)

        output = self.classifier(combined)
        return output


def save_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
        'model_type': 'Hybrid3Model'
    }
    torch.save(checkpoint, filepath)
    print(f'✅ Checkpoint saved to {filepath}')


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path='training_history_hybrid3.png'):
    """Plot and save training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(train_losses, label='Train Loss', marker='o')
    ax1.plot(val_losses, label='Val Loss', marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss (Hybrid 3-Model)')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(train_accs, label='Train Accuracy', marker='o')
    ax2.plot(val_accs, label='Val Accuracy', marker='o')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy (Hybrid 3-Model)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f'📊 Training history plot saved to {save_path}')


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch with gradient accumulation for effective larger batch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Gradient accumulation to simulate larger effective batch size
    accumulation_steps = 4  # Effective batch = BATCH_SIZE * 4 = 8

    optimizer.zero_grad()
    progress_bar = tqdm(dataloader, desc='Training')
    for step, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss = loss / accumulation_steps  # Normalize loss for accumulation
        loss.backward()

        # Update weights every accumulation_steps
        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * accumulation_steps  # Un-normalize for logging
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        progress_bar.set_postfix({
            'loss': f'{running_loss / (step + 1):.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })

    return running_loss / len(dataloader), 100. * correct / total


def validate(model, dataloader, criterion, device):
    """Validate with per-class accuracy logging."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # Track per-class accuracy
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Validation')
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Per-class stats
            c = (predicted == labels)
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

            progress_bar.set_postfix({
                'loss': f'{running_loss / len(dataloader):.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

    # Print per-class accuracy
    print("\n--- Validation Class Accuracy ---")
    for i in range(NUM_CLASSES):
        if class_total[i] > 0:
            print(f"  {CLASS_NAMES[i]}: {100 * class_correct[i] / class_total[i]:.2f}%")

    return running_loss / len(dataloader), 100. * correct / total


def main():
    print("=" * 60)
    print("🔬 HYBRID 3-MODEL TRAINING")
    print("   DenseNet121 (Local) + ViT-Base-384 (Global) + Swin-Base-384 (Hierarchical)")
    print("   Combining ALL Three Architectures for Maximum Accuracy")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Batch Size: {BATCH_SIZE} (effective: {BATCH_SIZE * 4} with gradient accumulation)")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Early Stopping Patience: {PATIENCE}")
    print(f"Model Save Path: {BEST_MODEL_PATH}")
    print("=" * 60)

    # Load Data
    print("\n📁 Loading datasets...")
    train_dataset = OralCancerDataset(
        TRAIN_DIR,
        transform=get_transforms(train=True, img_size=IMG_SIZE),
        extra_normal_aug=True
    )
    val_dataset = OralCancerDataset(
        VAL_DIR,
        transform=get_transforms(train=False, img_size=IMG_SIZE),
        extra_normal_aug=False
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True  # Prevent BatchNorm single-sample crash
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # Initialize Model
    print("\n🧠 Initializing Hybrid 3-Model...")
    model = Hybrid3Model(num_classes=NUM_CLASSES, pretrained=True).to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # Cosine annealing with warm restarts — good for complex multi-backbone models
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,       # Restart every 5 epochs
        T_mult=2,    # Double the period after each restart
        eta_min=1e-7
    )

    # Training tracking
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    patience_counter = 0

    print("\n🚀 Starting Training...\n")

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 50)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}   | Val Acc: {val_acc:.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")

        scheduler.step()

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, BEST_MODEL_PATH)
            patience_counter = 0
            print(f"  ✓ Best model saved! Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{PATIENCE}")

        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\n⚠️ Early stopping triggered after {epoch + 1} epochs")
            break

        print()

    # Save training history plot
    plot_training_history(train_losses, val_losses, train_accs, val_accs)

    print("=" * 60)
    print("🎉 Training Complete!")
    print(f"   Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"   Model saved to: {BEST_MODEL_PATH}")
    print("=" * 60)


if __name__ == '__main__':
    main()
