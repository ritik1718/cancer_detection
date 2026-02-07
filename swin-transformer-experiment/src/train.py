import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from src.config import Config
from src.dataset import OralCancerDataset, get_transforms
from src.model import get_model


def save_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }
    torch.save(checkpoint, filepath)
    print(f'✅ Checkpoint saved to {filepath}')


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path='training_history.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(train_losses, label='Train Loss', marker='o')
    ax1.plot(val_losses, label='Val Loss', marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(train_accs, label='Train Accuracy', marker='o')
    ax2.plot(val_accs, label='Val Accuracy', marker='o')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f'Training history plot saved to {save_path}')


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
        
        progress_bar.set_postfix({'loss': running_loss/len(dataloader), 
                                  'acc': 100.*correct/total})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Track per-class accuracy
    class_correct = [0] * Config.NUM_CLASSES
    class_total = [0] * Config.NUM_CLASSES
    
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
                
            progress_bar.set_postfix({'loss': running_loss/len(dataloader), 
                                      'acc': 100.*correct/total})
    
    print("\n--- Validation Class Accuracy ---")
    for i in range(Config.NUM_CLASSES):
        if class_total[i] > 0:
            print(f"Class {Config.CLASS_NAMES[i]}: {100 * class_correct[i] / class_total[i]:.2f}%")

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def main():
    print("=" * 50)
    print("Oral Cancer Detection - Heavy Class Weights Training")
    print("=" * 50)
    print(f"Device: {Config.DEVICE}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Epochs: {Config.NUM_EPOCHS}")
    print("=" * 50)
    
    # Load Data - DISABLED EXTRA AUGMENTATION
    print("\nLoading datasets...")
    train_dataset = OralCancerDataset(
        Config.TRAIN_DIR, 
        transform=get_transforms(train=True, img_size=Config.IMG_SIZE),
        extra_normal_aug=True  # ✅ ENABLED for aggressive Normal augmentation (but cleaner now)
    )
    val_dataset = OralCancerDataset(
        Config.VAL_DIR, 
        transform=get_transforms(train=False, img_size=Config.IMG_SIZE),
        extra_normal_aug=False
    )
    
    # Regular DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    # Setup Model
    print("\nInitializing model...")
    model = get_model(
    num_classes=Config.NUM_CLASSES, 
    pretrained=Config.PRETRAINED,
    model_name=Config.MODEL_NAME  # ✅ Add this
)
    
    # --- HEAVY MANUAL CLASS WEIGHTS ---
    # Normal is class 0, OSCC is class 1
    # --- CLASS WEIGHTS ---
    # Resetting to equal weights as the model should now be capable enough
    class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32).to(Config.DEVICE)
    
    print(f"USING MANUAL CLASS WEIGHTS: Normal={class_weights[0]:.1f}, OSCC={class_weights[1]:.1f}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.Adam(
        model.parameters(), 
        lr=Config.LEARNING_RATE, 
        weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        patience=5, 
        factor=0.5, 
        verbose=True
    )
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    patience_counter = 0
    
    print("\nStarting Training...\n")
    for epoch in range(Config.NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}")
        print("-" * 50)
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        val_loss, val_acc = validate(model, val_loader, criterion, Config.DEVICE)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, Config.BEST_MODEL_PATH)
            patience_counter = 0
            print(f"  ✓ Best model saved! Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{Config.PATIENCE}")
        
        if patience_counter >= Config.PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
        
        print()
    
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    print("=" * 50)
    print("Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {Config.BEST_MODEL_PATH}")
    print("=" * 50)


if __name__ == '__main__':
    main()
