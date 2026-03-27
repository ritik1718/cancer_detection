"""
ResNet Experiment Evaluation Script
=====================================
Evaluate any trained model from the resnet_Exp folder by passing the model name.

Usage:
    python resnet_Exp/evaluate.py resnet_solo
    python resnet_Exp/evaluate.py resnet_vit
    python resnet_Exp/evaluate.py axial_solo
    python resnet_Exp/evaluate.py axial_resnet
    python resnet_Exp/evaluate.py mobilenet_solo
    python resnet_Exp/evaluate.py axial_vit
    python resnet_Exp/evaluate.py mobilenet_vit
    python resnet_Exp/evaluate.py axial_densenet

The script will:
1. Dynamically import the model class from the corresponding training file
2. Load the saved checkpoint (e.g., resnet_solo_best.pth)
3. Evaluate on the test set with full metrics
"""

import os
import sys
import argparse
import importlib.util
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Add parent dir to path for src imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import OralCancerDataset, get_transforms

# ============================================
# CONFIGURATION
# ============================================
IMG_SIZE = 256
BATCH_SIZE = 4
NUM_CLASSES = 2
CLASS_NAMES = ['Normal', 'OSCC']

DATA_DIR = r'c:\Projects\oral-cancer-detection\data\raw'
TEST_DIR = os.path.join(DATA_DIR, 'test')

EXPERIMENT_DIR = r'c:\Projects\oral-cancer-detection\resnet_Exp'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Mapping: model name -> (training file, model class name)
MODEL_MAP = {
    'resnet_solo':    ('resnet_solo.py',    'ResNetSolo'),
    'resnet_vit':     ('resnet_vit.py',     'ResNetViTHybrid'),
    'axial_solo':     ('axial_solo.py',     'AxialSolo'),
    'axial_resnet':   ('axial_resnet.py',   'AxialResNetHybrid'),
    'mobilenet_solo': ('mobilenet_solo.py', 'MobileNetSolo'),
    'axial_vit':      ('axial_vit.py',      'AxialViTHybrid'),
    'mobilenet_vit':  ('mobilenet_vit.py',  'MobileNetViTHybrid'),
    'axial_densenet': ('axial_densenet.py', 'AxialDenseNetHybrid'),
    'densenet_resnet': ('densenet_resnet.py', 'DenseNetResNetHybrid'),
    'swin_transformer': ('swin_transformer.py', 'SwinTransformerSolo'),
    'resnet_swintransformer': ('resnet_swintransformer.py', 'ResNetSwinTransformerHybrid'),
    'densenet_swintransformer': ('densenet_swintransformer.py', 'DenseNetSwinTransformerHybrid'),
}


def load_model_class(model_name):
    """Dynamically import the model class from the corresponding training file."""
    if model_name not in MODEL_MAP:
        print(f"\n❌ Unknown model: '{model_name}'")
        print(f"\nAvailable models:")
        for name in sorted(MODEL_MAP.keys()):
            pth_path = os.path.join(EXPERIMENT_DIR, f'{name}_best.pth')
            status = '✅' if os.path.exists(pth_path) else '⬜'
            print(f"  {status} {name}")
        sys.exit(1)

    train_file, class_name = MODEL_MAP[model_name]
    train_path = os.path.join(EXPERIMENT_DIR, train_file)

    if not os.path.exists(train_path):
        print(f"\n❌ Training file not found: {train_path}")
        sys.exit(1)

    # Dynamically import the module
    spec = importlib.util.spec_from_file_location(f"resnet_module_{model_name}", train_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get the model class by name
    if hasattr(module, class_name):
        return getattr(module, class_name)
    else:
        print(f"\n❌ '{class_name}' class not found in {train_file}")
        sys.exit(1)


def evaluate_model(model, dataloader, device):
    """Run evaluation and collect predictions + probabilities."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Evaluating')
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress_bar.set_postfix({'acc': f'{100. * correct / total:.2f}%'})

    accuracy = 100. * correct / total
    return np.array(all_preds), np.array(all_labels), np.array(all_probs), accuracy


def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    save_path = os.path.join(EXPERIMENT_DIR, f'{model_name}_confusion_matrix.png')
    plt.savefig(save_path)
    plt.close()
    print(f'📊 Confusion matrix saved to {save_path}')


def print_classification_report(y_true, y_pred, class_names):
    """Print detailed classification report."""
    print("\n" + "=" * 50)
    print("Classification Report")
    print("=" * 50)
    print(classification_report(y_true, y_pred, target_names=class_names))


def print_per_class_accuracy(y_true, y_pred, class_names):
    """Print per-class accuracy breakdown."""
    cm = confusion_matrix(y_true, y_pred)
    print("\n--- Per-Class Accuracy ---")
    for i, name in enumerate(class_names):
        if cm[i].sum() > 0:
            acc = 100.0 * cm[i][i] / cm[i].sum()
            print(f"  {name}: {acc:.2f}% ({cm[i][i]}/{cm[i].sum()})")


def print_confidence_stats(all_probs, all_labels, all_preds, class_names):
    """Print prediction confidence statistics."""
    print("\n--- Prediction Confidence Statistics ---")
    all_probs = np.array(all_probs)

    # Overall confidence
    max_probs = np.max(all_probs, axis=1)
    print(f"  Overall mean confidence: {np.mean(max_probs) * 100:.2f}%")
    print(f"  Overall min  confidence: {np.min(max_probs) * 100:.2f}%")

    # Correct vs wrong confidence
    correct_mask = (np.array(all_preds) == np.array(all_labels))
    if correct_mask.sum() > 0:
        print(f"  Correct predictions mean confidence: {np.mean(max_probs[correct_mask]) * 100:.2f}%")
    if (~correct_mask).sum() > 0:
        print(f"  Wrong predictions mean confidence:   {np.mean(max_probs[~correct_mask]) * 100:.2f}%")

    # Per-class confidence
    for i, name in enumerate(class_names):
        mask = (np.array(all_labels) == i)
        if mask.sum() > 0:
            class_probs = max_probs[mask]
            print(f"  {name} mean confidence: {np.mean(class_probs) * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a ResNet experiment model on the test set.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python resnet_Exp/evaluate.py resnet_solo
  python resnet_Exp/evaluate.py resnet_vit
  python resnet_Exp/evaluate.py axial_solo
  python resnet_Exp/evaluate.py axial_resnet
  python resnet_Exp/evaluate.py mobilenet_solo
  python resnet_Exp/evaluate.py axial_vit
  python resnet_Exp/evaluate.py mobilenet_vit
  python resnet_Exp/evaluate.py swin_transformer
  python resnet_Exp/evaluate.py resnet_swintransformer
  python resnet_Exp/evaluate.py densenet_swintransformer
        """
    )
    parser.add_argument(
        'model_name',
        type=str,
        help='Name of the model to evaluate (e.g., resnet_solo, resnet_vit, axial_solo, axial_resnet, mobilenet_solo, axial_vit, mobilenet_vit, axial_densenet, swin_transformer, resnet_swintransformer, densenet_swintransformer)'
    )
    args = parser.parse_args()

    model_name = args.model_name
    checkpoint_path = os.path.join(EXPERIMENT_DIR, f'{model_name}_best.pth')

    print("=" * 60)
    print(f"🔬 RESNET EXPERIMENT EVALUATION: {model_name}")
    print("=" * 60)
    print(f"  Device:     {DEVICE}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Test Dir:   {TEST_DIR}")
    print("=" * 60)

    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"\n❌ Checkpoint not found: {checkpoint_path}")
        train_file = MODEL_MAP.get(model_name, ('???',))[0]
        print(f"   Train this model first using: python resnet_Exp/{train_file}")
        sys.exit(1)

    # Load test dataset
    print("\n📁 Loading test dataset...")
    test_dataset = OralCancerDataset(
        TEST_DIR,
        transform=get_transforms(train=False, img_size=IMG_SIZE)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Test batches: {len(test_loader)}")

    # Dynamically load the correct model class
    train_file, class_name = MODEL_MAP[model_name]
    print(f"\n🧠 Loading model architecture '{class_name}' from {train_file}...")
    ModelClass = load_model_class(model_name)
    model = ModelClass(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)

    # Load trained weights
    print(f"📦 Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Print checkpoint info
    if 'epoch' in checkpoint:
        print(f"  Checkpoint epoch: {checkpoint['epoch'] + 1}")
    if 'accuracy' in checkpoint:
        print(f"  Checkpoint val accuracy: {checkpoint['accuracy']:.2f}%")
    if 'experiment' in checkpoint:
        print(f"  Experiment tag: {checkpoint['experiment']}")
    print("  ✅ Model loaded successfully\n")

    # Evaluate
    predictions, labels, probs, accuracy = evaluate_model(model, test_loader, DEVICE)

    # Results
    print("\n" + "=" * 60)
    print(f"🎯 Test Accuracy: {accuracy:.2f}%")
    print("=" * 60)

    print_per_class_accuracy(labels, predictions, CLASS_NAMES)
    print_confidence_stats(probs, labels, predictions, CLASS_NAMES)
    print_classification_report(labels, predictions, CLASS_NAMES)
    plot_confusion_matrix(labels, predictions, CLASS_NAMES, model_name)

    print("\n" + "=" * 60)
    print(f"✅ Evaluation of '{model_name}' Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
