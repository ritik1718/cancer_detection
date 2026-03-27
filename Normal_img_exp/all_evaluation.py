"""
Mass Evaluation Aggregator Script
===================================
Evaluates all trained models in the experiment directory and displays
the combined results (Accuracy, Sensitivity, Specificity, F1-Score)
in a side-by-side tabular format.

Usage:
    python Normal_img_exp/all_evaluation.py
"""

import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Add parent dir to path for src imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, os.path.dirname(_SCRIPT_DIR))

# Import logic from the existing evaluate.py script
from evaluate import MODEL_MAP, EXPERIMENT_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE, NUM_CLASSES, DEVICE, CLASS_NAMES
from evaluate import load_model_class, evaluate_model
from src.dataset import OralCancerDataset, get_transforms


def main():
    print("=" * 90)
    print("🔬 RUNNING FULL MODEL EVALUATION BENCHMARK")
    print("=" * 90)
    
    # Load test dataset once for all models to ensure entirely fair comparison
    print("\n📁 Loading unified test dataset...")
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
    
    results = []
    
    for model_name, (train_file, class_name) in MODEL_MAP.items():
        checkpoint_path = os.path.join(EXPERIMENT_DIR, f'{model_name}_best.pth')
        
        if not os.path.exists(checkpoint_path):
            results.append({
                'name': model_name,
                'status': 'Not Trained',
                'acc': 0, 'sens': 0, 'spec': 0, 'f1': 0
            })
            continue
            
        print(f"\n🧠 Evaluating {model_name}...")
        
        try:
            # Dynamically load the correct class
            ModelClass = load_model_class(model_name)
            model = ModelClass(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
            
            # Load weights
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])    
            
            # Run inference loop
            predictions, labels, probs, accuracy = evaluate_model(model, test_loader, DEVICE)
            
            # Calculate Custom Metrics
            cm = confusion_matrix(labels, predictions)
            
            # cm[0,0]: True Negative (Normal correct)
            # cm[0,1]: False Positive (Normal predicted as OSCC)
            # cm[1,0]: False Negative (OSCC predicted as Normal)
            # cm[1,1]: True Positive (OSCC correct)
            
            # Depending on sklearn version and number of classes present in y_pred, ravel() might differ,
            # but since we strictly have 2 classes in TEST_DIR, it will be 2x2.
            # Handle edge case if a model predicts perfectly only one class on a tiny batch.
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                tn = cm[0,0] if len(cm) > 0 else 0
                tp = cm[1,1] if len(cm) > 1 else 0
                fp = fn = 0
            
            sensitivity = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0  # Recall for OSCC
            specificity = (tn / (tn + fp)) * 100 if (tn + fp) > 0 else 0  # Recall for Normal
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results.append({
                'name': model_name,
                'status': 'Evaluated',
                'acc': accuracy,
                'sens': sensitivity,
                'spec': specificity,
                'f1': f1 * 100
            })
            print(f"  ✅ Acc: {accuracy:.2f}%, Sens: {sensitivity:.2f}%, Spec: {specificity:.2f}%, F1: {f1*100:.2f}%")
            
        except Exception as e:
            print(f"  ❌ Error evaluating {model_name}: {e}")
            results.append({
                'name': model_name,
                'status': 'Error',
                'acc': 0, 'sens': 0, 'spec': 0, 'f1': 0
            })
            
            
    # ============================================
    # PRINT TABULAR RESULTS
    # ============================================
    print("\n\n" + "=" * 90)
    print(f"🏆 FINAL MODEL BENCHMARK RESULTS")
    print("=" * 90)
    print(f"{'MODEL NAME':<28} | {'STATUS':<12} | {'ACCURACY':<9} | {'SENSITIVITY':<11} | {'SPECIFICITY':<11} | {'F1-SCORE':<9}")
    print("-" * 90)
    
    # Sort results by accuracy descending
    results.sort(key=lambda x: x['acc'], reverse=True)
    
    for r in results:
        if r['status'] == 'Evaluated':
            print(f"{r['name']:<28} | {r['status']:<12} | {r['acc']:>8.2f}% | {r['sens']:>10.2f}% | {r['spec']:>10.2f}% | {r['f1']:>8.2f}%")
        else:
            print(f"{r['name']:<28} | {r['status']:<12} | {'-':>9} | {'-':>11} | {'-':>11} | {'-':>9}")
            
    print("=" * 90)
    print("* Sensitivity = True Positive Rate for OSCC (correctly identified cancers)")
    print("* Specificity = True Negative Rate for Normal (correctly identified healthy)")
    print("* F1-Score    = Harmonic mean of precision and sensitivity for OSCC")
    print("=" * 90 + "\n")

if __name__ == '__main__':
    main()
