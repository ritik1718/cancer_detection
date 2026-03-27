"""
Dataset Preparation Script
============================
Downloads two Kaggle datasets, merges them into binary Normal/OSCC classes,
applies offline augmentation to balance classes, and splits into train/val/test.

Datasets:
  1. javedrashid/mouth-and-oral-diseases-mod (MOD)
     - 7 classes: Gum, CaS, CoS, OLP, OT, MC, OC
     - MC + OC -> OSCC; rest -> Normal
  2. shivam17299/oral-cancer-lips-and-tongue-images
     - Cancerous/Non-Cancerous lips and tongue
     - Cancerous -> OSCC; Non-Cancerous -> Normal

Usage:
    python Normal_img_exp/prepare_data.py
"""

import os
import sys
import shutil
import zipfile
import subprocess
import random
import glob
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from collections import Counter

# ============================================
# CONFIGURATION
# ============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data', 'raw')
BACKUP_DIR = os.path.join(SCRIPT_DIR, 'data', 'raw_backup')
DOWNLOAD_DIR = os.path.join(SCRIPT_DIR, 'kaggle_downloads')

# Kaggle dataset slugs
DATASETS = [
    'javedrashid/mouth-and-oral-diseases-mod',
    'shivam17299/oral-cancer-lips-and-tongue-images',
]

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Random seed for reproducibility
SEED = 42

# Class mapping keywords (case-insensitive substring match on folder names)
OSCC_KEYWORDS = ['cancer', 'mc', 'oc', 'oscc', 'tumor', 'malignant']
NORMAL_KEYWORDS = ['normal', 'healthy', 'non-cancer', 'non_cancer', 'noncancer',
                   'gum', 'gingivostomatitis', 'canker', 'cold sore', 'coldsore',
                   'cos', 'cas', 'lichen', 'olp', 'thrush', 'ot',
                   'non-cancerous', 'non_cancerous', 'noncancerous', 'benign']

# Augmentation multiplier for minority class
AUG_TARGET_RATIO = 1.0  # Target 1:1 ratio between classes

# Valid image extensions
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


# ============================================
# DOWNLOAD FUNCTIONS
# ============================================

def download_datasets():
    """Download datasets from Kaggle using the Kaggle Python API."""
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        print(f"  ❌ Kaggle API authentication failed: {e}")
        print("     Set up your API key: https://www.kaggle.com/docs/api")
        sys.exit(1)

    for slug in DATASETS:
        dataset_name = slug.split('/')[-1]
        target_dir = os.path.join(DOWNLOAD_DIR, dataset_name)

        if os.path.exists(target_dir) and any(os.scandir(target_dir)):
            print(f"  ✅ Already downloaded: {dataset_name}")
            continue

        print(f"  ⬇️  Downloading: {slug} ...")
        try:
            api.dataset_download_files(slug, path=DOWNLOAD_DIR, unzip=True)
            print(f"  ✅ Downloaded: {dataset_name}")
        except Exception as e:
            print(f"  ❌ Failed to download {slug}: {e}")
            sys.exit(1)


def extract_zips():
    """Extract any remaining zip files in the download directory."""
    for zip_file in glob.glob(os.path.join(DOWNLOAD_DIR, '*.zip')):
        dataset_name = os.path.splitext(os.path.basename(zip_file))[0]
        extract_dir = os.path.join(DOWNLOAD_DIR, dataset_name)

        if os.path.exists(extract_dir) and any(os.scandir(extract_dir)):
            print(f"  ✅ Already extracted: {dataset_name}")
            continue

        print(f"  📦 Extracting: {os.path.basename(zip_file)} ...")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_file, 'r') as zf:
            zf.extractall(extract_dir)
        print(f"  ✅ Extracted to: {extract_dir}")


# ============================================
# CLASS MAPPING
# ============================================

def classify_folder(folder_name):
    """Determine if a folder contains OSCC or Normal images based on its name."""
    name_lower = folder_name.lower().strip()

    # Check for OSCC first (cancer-related keywords)
    # But be careful: "non-cancerous" should NOT match as OSCC
    if any(neg in name_lower for neg in ['non-cancer', 'non_cancer', 'noncancer', 'non cancerous']):
        return 'Normal'

    if any(kw in name_lower for kw in OSCC_KEYWORDS):
        return 'OSCC'

    # Check for Normal keywords
    if any(kw in name_lower for kw in NORMAL_KEYWORDS):
        return 'Normal'

    # Unknown - skip with warning
    return None


def collect_images_from_dir(root_dir):
    """Recursively find all image files and classify them into Normal/OSCC."""
    normal_images = []
    oscc_images = []
    skipped_folders = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Get image files in this directory
        images = [f for f in filenames if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS]

        if not images:
            continue

        # Use the immediate parent folder name for classification
        folder_name = os.path.basename(dirpath)
        classification = classify_folder(folder_name)

        if classification == 'OSCC':
            for img in images:
                oscc_images.append(os.path.join(dirpath, img))
        elif classification == 'Normal':
            for img in images:
                normal_images.append(os.path.join(dirpath, img))
        else:
            skipped_folders.append((folder_name, len(images)))

    return normal_images, oscc_images, skipped_folders


# ============================================
# AUGMENTATION
# ============================================

def augment_image(image):
    """Apply random augmentations to a PIL Image."""
    augmented = image.copy()

    # Random horizontal flip
    if random.random() > 0.5:
        augmented = augmented.transpose(Image.FLIP_LEFT_RIGHT)

    # Random vertical flip
    if random.random() > 0.5:
        augmented = augmented.transpose(Image.FLIP_TOP_BOTTOM)

    # Random rotation
    angle = random.choice([0, 90, 180, 270]) + random.uniform(-15, 15)
    augmented = augmented.rotate(angle, resample=Image.BICUBIC, fillcolor=(0, 0, 0))

    # Random brightness
    if random.random() > 0.3:
        factor = random.uniform(0.7, 1.3)
        augmented = ImageEnhance.Brightness(augmented).enhance(factor)

    # Random contrast
    if random.random() > 0.3:
        factor = random.uniform(0.7, 1.3)
        augmented = ImageEnhance.Contrast(augmented).enhance(factor)

    # Random saturation
    if random.random() > 0.4:
        factor = random.uniform(0.7, 1.3)
        augmented = ImageEnhance.Color(augmented).enhance(factor)

    # Random sharpness
    if random.random() > 0.5:
        factor = random.uniform(0.5, 2.0)
        augmented = ImageEnhance.Sharpness(augmented).enhance(factor)

    return augmented


def apply_offline_augmentation(image_paths, target_count, output_dir, class_name):
    """
    Augment images offline until we reach target_count total images.
    Copies originals + generates augmented versions.
    """
    os.makedirs(output_dir, exist_ok=True)
    current_count = 0

    # First, copy all original images
    for src_path in image_paths:
        ext = os.path.splitext(src_path)[1]
        dst_name = f"{class_name}_orig_{current_count:05d}{ext}"
        dst_path = os.path.join(output_dir, dst_name)
        try:
            img = Image.open(src_path).convert('RGBA').convert('RGB')
            img.save(dst_path)
            current_count += 1
        except Exception as e:
            print(f"    ⚠️ Skipped corrupt image: {src_path} ({e})")

    original_count = current_count
    print(f"    📁 Copied {original_count} original {class_name} images")

    # Then augment until we reach target
    if current_count < target_count:
        aug_needed = target_count - current_count
        print(f"    🔄 Generating {aug_needed} augmented {class_name} images...")

        aug_idx = 0
        while current_count < target_count:
            src_path = image_paths[aug_idx % len(image_paths)]
            try:
                img = Image.open(src_path).convert('RGBA').convert('RGB')
                aug_img = augment_image(img)
                ext = os.path.splitext(src_path)[1]
                dst_name = f"{class_name}_aug_{current_count:05d}{ext}"
                dst_path = os.path.join(output_dir, dst_name)
                aug_img.save(dst_path)
                current_count += 1
            except Exception as e:
                pass  # Skip corrupt images silently during augmentation
            aug_idx += 1

    return current_count


# ============================================
# SPLIT FUNCTION
# ============================================

def split_data(merged_dir, output_dir):
    """Split merged data into train/val/test with stratification."""
    random.seed(SEED)

    for class_name in ['Normal', 'OSCC']:
        class_dir = os.path.join(merged_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"    ⚠️ Missing class directory: {class_dir}")
            continue

        images = [f for f in os.listdir(class_dir)
                  if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS]
        random.shuffle(images)

        n = len(images)
        train_end = int(n * TRAIN_RATIO)
        val_end = train_end + int(n * VAL_RATIO)

        splits = {
            'train': images[:train_end],
            'val': images[train_end:val_end],
            'test': images[val_end:]
        }

        for split_name, split_images in splits.items():
            split_class_dir = os.path.join(output_dir, split_name, class_name)
            os.makedirs(split_class_dir, exist_ok=True)

            for img_name in split_images:
                src = os.path.join(class_dir, img_name)
                dst = os.path.join(split_class_dir, img_name)
                shutil.copy2(src, dst)

            print(f"    {split_name}/{class_name}: {len(split_images)} images")


# ============================================
# MAIN
# ============================================

def main():
    print("=" * 60)
    print("🔬 ORAL CANCER DETECTION - DATA PREPARATION")
    print("=" * 60)

    # Step 1: Download datasets
    print("\n📥 Step 1: Downloading Kaggle datasets...")
    download_datasets()
    extract_zips()

    # Step 2: Collect and classify images
    print("\\n📂 Step 2: Scanning and classifying downloaded images...")
    
    print(f"\\n  Scanning download directory: {DOWNLOAD_DIR}")
    all_normal, all_oscc, skipped = collect_images_from_dir(DOWNLOAD_DIR)
    
    print(f"    Normal: {len(all_normal)}, OSCC: {len(all_oscc)}")

    if skipped:
        for folder, count in skipped:
            print(f"    ⚠️ Skipped folder '{folder}' ({count} images) - unknown class")

    print(f"\n  📊 Total collected:")
    print(f"    Normal: {len(all_normal)}")
    print(f"    OSCC:   {len(all_oscc)}")

    if len(all_normal) == 0 or len(all_oscc) == 0:
        print("  ❌ One or both classes have 0 images. Check dataset structure.")
        print("  Listing download directory contents for debugging:")
        for dirpath, dirnames, filenames in os.walk(DOWNLOAD_DIR):
            level = dirpath.replace(DOWNLOAD_DIR, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"  {indent}{os.path.basename(dirpath)}/")
            if level < 3:  # Only show first 3 levels
                subindent = ' ' * 2 * (level + 1)
                for f in filenames[:5]:
                    print(f"  {subindent}{f}")
                if len(filenames) > 5:
                    print(f"  {subindent}... and {len(filenames) - 5} more files")
        sys.exit(1)

    # Step 3: Augment to balance classes
    print("\n🔄 Step 3: Augmenting and balancing classes...")
    merged_dir = os.path.join(SCRIPT_DIR, 'data', 'merged_temp')

    # Clean up any previous temp dir
    if os.path.exists(merged_dir):
        shutil.rmtree(merged_dir)

    max_count = max(len(all_normal), len(all_oscc))
    target_count = int(max_count * AUG_TARGET_RATIO)

    # Force target_count to be 2500 per class (to get 5000 images total)
    target_count = 2500

    print(f"  Target count per class: {target_count}")

    normal_count = apply_offline_augmentation(
        all_normal, target_count,
        os.path.join(merged_dir, 'Normal'), 'Normal'
    )
    oscc_count = apply_offline_augmentation(
        all_oscc, target_count,
        os.path.join(merged_dir, 'OSCC'), 'OSCC'
    )

    print(f"\n  📊 After augmentation:")
    print(f"    Normal: {normal_count}")
    print(f"    OSCC:   {oscc_count}")

    # Step 4: Backup existing data
    print("\n💾 Step 4: Backing up existing data...")
    if os.path.exists(DATA_DIR):
        if os.path.exists(BACKUP_DIR):
            shutil.rmtree(BACKUP_DIR)
        shutil.copytree(DATA_DIR, BACKUP_DIR)
        print(f"  ✅ Backed up to: {BACKUP_DIR}")
        shutil.rmtree(DATA_DIR)
    else:
        print("  ℹ️ No existing data to backup")

    # Step 5: Split into train/val/test
    print("\n✂️ Step 5: Splitting into train/val/test...")
    os.makedirs(DATA_DIR, exist_ok=True)
    split_data(merged_dir, DATA_DIR)

    # Cleanup temp merged dir
    shutil.rmtree(merged_dir)

    # Step 6: Print summary
    print("\n" + "=" * 60)
    print("✅ DATA PREPARATION COMPLETE")
    print("=" * 60)

    for split in ['train', 'val', 'test']:
        for cls in ['Normal', 'OSCC']:
            cls_dir = os.path.join(DATA_DIR, split, cls)
            if os.path.exists(cls_dir):
                count = len([f for f in os.listdir(cls_dir)
                           if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS])
                print(f"  {split}/{cls}: {count} images")

    print(f"\n  Output directory: {DATA_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
