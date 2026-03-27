"""
Run All Experiments
=====================
Sequentially runs all 12 training scripts in Normal_img_exp/.
Each script uses its own training parameters as defined in the file.
Scripts are ALWAYS re-run (retrained) regardless of existing checkpoints.

Usage:
    python Normal_img_exp/run_all.py
    python Normal_img_exp/run_all.py --skip 3     # Skip first N experiments
"""

import os
import sys
import subprocess
import time
import argparse

# ============================================
# CONFIGURATION
# ============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# Training scripts in execution order (solo models first, then hybrids)
EXPERIMENTS = [
    ('resnet_solo.py',              'ResNet50 Solo'),
    ('mobilenet_solo.py',           'MobileNetV2 Solo'),
    ('axial_solo.py',               'Axial Attention Solo'),
    ('swin_transformer.py',         'Swin Transformer V2 Solo'),
    ('resnet_vit.py',               'ResNet50 + ViT Hybrid'),
    ('mobilenet_vit.py',            'MobileNetV2 + ViT Hybrid'),
    ('axial_vit.py',                'Axial + ViT Hybrid'),
    ('axial_resnet.py',             'Axial + ResNet50 Hybrid'),
    ('axial_densenet.py',           'Axial + DenseNet121 Hybrid'),
    ('densenet_resnet.py',          'DenseNet121 + ResNet50 Hybrid'),
    ('resnet_swintransformer.py',   'ResNet50 + SwinV2 Hybrid'),
    ('densenet_swintransformer.py', 'DenseNet121 + SwinV2 Hybrid'),
]


def get_experiment_name(script_file):
    """Extract experiment name from script filename."""
    return os.path.splitext(script_file)[0]


def run_experiment(script_file, description, index, total):
    """Run a single training script."""
    script_path = os.path.join(SCRIPT_DIR, script_file)
    experiment_name = get_experiment_name(script_file)

    print(f"\n{'=' * 70}")
    print(f"  [{index}/{total}] {description}")
    print(f"  Script: {script_file}")
    print(f"  Experiment: {experiment_name}")
    print(f"{'=' * 70}")

    if not os.path.exists(script_path):
        print(f"  ❌ Script not found: {script_path}")
        return False

    # Delete existing checkpoint so the model trains fresh
    pth_path = os.path.join(SCRIPT_DIR, f'{experiment_name}_best.pth')
    if os.path.exists(pth_path):
        os.remove(pth_path)
        print(f"  🗑️ Removed old checkpoint: {experiment_name}_best.pth")

    start_time = time.time()

    try:
        # Run the training script as a subprocess
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=SCRIPT_DIR,
            check=True,
        )
        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        print(f"\n  ✅ {description} completed in {minutes}m {seconds}s")
        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        print(f"\n  ❌ {description} FAILED after {minutes}m {seconds}s (exit code: {e.returncode})")
        return False

    except KeyboardInterrupt:
        print(f"\n  ⚠️ Training interrupted by user")
        raise


def main():
    parser = argparse.ArgumentParser(description='Run all training experiments sequentially.')
    parser.add_argument('--skip', type=int, default=0,
                        help='Skip the first N experiments')
    args = parser.parse_args()

    print("=" * 70)
    print("🔬 ORAL CANCER DETECTION - RUN ALL EXPERIMENTS")
    print("=" * 70)
    print(f"  Total experiments: {len(EXPERIMENTS)}")
    print(f"  Mode:              ALWAYS RETRAIN (fresh training every run)")
    print(f"  Skip first:        {args.skip}")
    print(f"  Script dir:        {SCRIPT_DIR}")
    print("=" * 70)

    # Show experiment list
    print("\n📋 Experiment Queue:")
    for i, (script, desc) in enumerate(EXPERIMENTS, 1):
        skip_marker = ' [SKIP]' if i <= args.skip else ''
        print(f"  {i:2d}. {desc:40s} 🔄 will train{skip_marker}")

    # Run experiments
    results = {}
    total = len(EXPERIMENTS)
    start_total = time.time()

    for i, (script, desc) in enumerate(EXPERIMENTS, 1):
        if i <= args.skip:
            print(f"\n  ⏭️ Skipping [{i}/{total}] {desc}")
            results[desc] = 'SKIPPED'
            continue

        try:
            success = run_experiment(script, desc, i, total)
            results[desc] = '✅ SUCCESS' if success else '❌ FAILED'
        except KeyboardInterrupt:
            results[desc] = '⚠️ INTERRUPTED'
            print("\n\n  User interrupted. Stopping all experiments.")
            break

    # Print summary
    total_elapsed = time.time() - start_total
    total_minutes = int(total_elapsed // 60)
    total_seconds = int(total_elapsed % 60)

    print(f"\n\n{'=' * 70}")
    print("📊 EXPERIMENT SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total time: {total_minutes}m {total_seconds}s\n")

    for desc, status in results.items():
        print(f"  {desc:40s} {status}")

    print(f"\n{'=' * 70}")
    print("✅ All experiments complete!")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
