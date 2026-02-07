import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from src.config import Config
from src.dataset import OralCancerDataset, get_transforms
from src.model import get_model
from src.utils import plot_confusion_matrix, print_classification_report


def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Evaluating')
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({'acc': 100.*correct/total})
    
    accuracy = 100. * correct / total
    return np.array(all_preds), np.array(all_labels), accuracy


def main():
    print("=" * 50)
    print("Model Evaluation")
    print("=" * 50)
    print(f"Device: {Config.DEVICE}")
    print(f"Model: {Config.BEST_MODEL_PATH}")
    print("=" * 50)
    
    # Load test dataset
    test_dataset = OralCancerDataset(Config.TEST_DIR, transform=get_transforms(train=False, img_size=Config.IMG_SIZE))
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS
    )
    
    print(f"\nTest dataset size: {len(test_dataset)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Load model
    model =  get_model(
    num_classes=Config.NUM_CLASSES, 
    pretrained=False,
    model_name=Config.MODEL_NAME  # ✅ Add this
)
    
    checkpoint = torch.load(Config.BEST_MODEL_PATH, map_location=Config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded successfully\n")
    
    # Evaluate
    predictions, labels, accuracy = evaluate_model(model, test_loader, Config.DEVICE)
    
    print("\n" + "=" * 50)
    print(f"Test Accuracy: {accuracy:.2f}%")
    print("=" * 50)
    
    print_classification_report(labels, predictions, Config.CLASS_NAMES)
    plot_confusion_matrix(labels, predictions, Config.CLASS_NAMES)


if __name__ == '__main__':
    main()
