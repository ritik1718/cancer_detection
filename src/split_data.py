import os
import shutil
import random
from glob import glob

def split_dataset():
    # Configuration
    DATA_ROOT = 'data/raw'
    CLASSES = ['Normal', 'OSCC']
    SPLIT_RATIO = {'train': 0.70, 'val': 0.15, 'test': 0.15}
    
    # 1. Collect all valid images
    all_images = {'Normal': [], 'OSCC': []}
    
    print("Scanning for images...")
    # Look in the existing train/val/test folders or just root if flat
    for class_name in CLASSES:
        # Search recursively to find all images of this class regardless of current split
        pattern = os.path.join(DATA_ROOT, '**', class_name, '*')
        files = glob(pattern, recursive=True)
        
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        for f in files:
            if f.lower().endswith(valid_extensions):
                all_images[class_name].append(f)
        
        print(f"Found {len(all_images[class_name])} existing {class_name} images.")

    # 2. Clear old directories (but keep images in memory/temp list)
    # Actually, to be safe, we won't delete until we move. 
    # But we need fresh target folders.
    
    # Create temp structure to avoid overwriting issues if source=dest
    TEMP_DIR = 'data/temp_split'
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
        
    for split in ['train', 'val', 'test']:
        for class_name in CLASSES:
            os.makedirs(os.path.join(TEMP_DIR, split, class_name), exist_ok=True)
            
    # 3. Shuffle and Distribute
    print("\nDistributing files...")
    stats = {s: {c: 0 for c in CLASSES} for s in SPLIT_RATIO}
    
    for class_name in CLASSES:
        images = all_images[class_name]
        random.shuffle(images)
        
        n_total = len(images)
        n_train = int(n_total * SPLIT_RATIO['train'])
        n_val = int(n_total * SPLIT_RATIO['val'])
        # Test gets the rest to ensure sum is total
        n_test = n_total - n_train - n_val
        
        # Split list
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train+n_val]
        test_imgs = images[n_train+n_val:]
        
        # Move function
        def copy_files(file_list, split_name):
            print(f"  Moving {len(file_list)} files to {split_name}/{class_name}...")
            for src_path in file_list:
                filename = os.path.basename(src_path)
                # Handle duplicate filenames by prefixing if needed (unlikely here but good practice)
                dst_path = os.path.join(TEMP_DIR, split_name, class_name, filename)
                counter = 1
                while os.path.exists(dst_path):
                    name, ext = os.path.splitext(filename)
                    dst_path = os.path.join(TEMP_DIR, split_name, class_name, f"{name}_{counter}{ext}")
                    counter += 1
                
                shutil.copy2(src_path, dst_path)
                stats[split_name][class_name] += 1

        copy_files(train_imgs, 'train')
        copy_files(val_imgs, 'val')
        copy_files(test_imgs, 'test')

    # 4. Replace old 'raw' contents with 'temp_split' contents
    # We will backup the old structure just in case? No, user wants changes.
    # But we must be careful not to delete 'data/raw' if it contains the source files we are copying!
    # Since we copied to data/temp_split, we can now safely clear data/raw/{train,val,test} and move temp_split there.
    
    print("\nFinalizing directory structure...")
    
    # Remove old train/val/test
    for split in ['train', 'val', 'test']:
        old_split_path = os.path.join(DATA_ROOT, split)
        if os.path.exists(old_split_path):
            shutil.rmtree(old_split_path)
    
    # Move from temp to raw
    for split in ['train', 'val', 'test']:
        shutil.move(os.path.join(TEMP_DIR, split), os.path.join(DATA_ROOT, split))
        
    os.rmdir(TEMP_DIR)
    
    print("="*40)
    print("Final Split Statistics")
    print("="*40)
    total_files = sum(len(v) for v in all_images.values())
    
    print(f"{'Split':<10} | {'Normal':<8} | {'OSCC':<8} | {'Total':<8} | {'%':<6}")
    print("-" * 50)
    
    for split in ['train', 'val', 'test']:
        n_norm = stats[split]['Normal']
        n_oscc = stats[split]['OSCC']
        n_tot = n_norm + n_oscc
        pct = (n_tot / total_files) * 100 if total_files > 0 else 0
        print(f"{split:<10} | {n_norm:<8} | {n_oscc:<8} | {n_tot:<8} | {pct:.1f}%")
        
    print("-" * 50)
    print("✅ Data split completed successfully.")

if __name__ == '__main__':
    split_dataset()
