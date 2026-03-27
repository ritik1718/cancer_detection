
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.dataset import OralCancerDataset, get_transforms

# ============================================
# EXPERIMENT 6: Low Learning Rate
# Description: Lowers learning rate to 1e-5.
# Comparison: Lower LR might help in fine-tuning pre-trained weights without destroying them, potentially leading to better minima.
# Architectures: DenseNet121 + ViT-Base (Baseline Concatenation)
# Learning Rate: 1e-5 (vs 1e-4 in baseline)
# ============================================

IMG_SIZE = 384
BATCH_SIZE = 4
NUM_EPOCHS = 30
LEARNING_RATE = 1e-5 # Lowered
WEIGHT_DECAY = 1e-4
PATIENCE = 8

DATA_DIR = r'c:\Projects\oral-cancer-detection\data\raw'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')

MODEL_SAVE_DIR = r'c:\Projects\oral-cancer-detection\experiment'
EXPERIMENT_NAME = 'exp6_lr_1e5'
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, f'{EXPERIMENT_NAME}_best.pth')

CLASS_NAMES = ['Normal', 'OSCC']
NUM_CLASSES = 2

class HybridViTDenseNet(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(HybridViTDenseNet, self).__init__()
        if pretrained:
            self.densenet = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        else:
            self.densenet = models.densenet121(weights=None)
        
        self.densenet_features = self.densenet.features
        self.densenet_pooling = nn.AdaptiveAvgPool2d((1, 1))
        densenet_dim = 1024
        
        self.vit = timm.create_model('vit_base_patch16_384', pretrained=pretrained, num_classes=0)
        vit_dim = 768
        
        self.fusion_dim = densenet_dim + vit_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        dn_feat = self.densenet_features(x)
        dn_feat = self.densenet_pooling(dn_feat)
        dn_feat = torch.flatten(dn_feat, 1)
        vit_feat = self.vit(x)
        combined = torch.cat((dn_feat, vit_feat), dim=1)
        return self.classifier(combined)

def save_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
        'experiment': EXPERIMENT_NAME
    }
    torch.save(checkpoint, filepath)
    print(f'✅ Checkpoint saved to {filepath}')

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    progress_bar = tqdm(dataloader, desc='Training')
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        progress_bar.set_postfix({'loss': f'{running_loss/len(dataloader):.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    return running_loss / len(dataloader), 100. * correct / total

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
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
            progress_bar.set_postfix({'loss': f'{running_loss/len(dataloader):.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    return running_loss / len(dataloader), 100. * correct / total

def main():
    print(f"🔬 EXPERIMENT START: {EXPERIMENT_NAME}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dataset = OralCancerDataset(TRAIN_DIR, transform=get_transforms(train=True, img_size=IMG_SIZE), extra_normal_aug=True)
    val_dataset = OralCancerDataset(VAL_DIR, transform=get_transforms(train=False, img_size=IMG_SIZE), extra_normal_aug=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    model = HybridViTDenseNet(num_classes=NUM_CLASSES).to(device)
    
    criterion = nn.CrossEntropyLoss()
    # Using the lower learning rate
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"  Summary: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        scheduler.step(val_loss)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, BEST_MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered")
                break
    print(f"Experiment {EXPERIMENT_NAME} Complete. Best Acc: {best_val_acc:.2f}%")

if __name__ == '__main__':
    main()
