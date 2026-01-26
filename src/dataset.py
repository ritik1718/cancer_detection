import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


class OralCancerDataset(Dataset):
    def __init__(self, data_dir, transform=None, extra_normal_aug=False):
        self.data_dir = data_dir
        self.transform = transform
        self.extra_normal_aug = extra_normal_aug
        self.classes = ['Normal', 'OSCC']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        # Load images from both Normal and OSCC folders
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                continue
                
            class_idx = self.class_to_idx[class_name]
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(class_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Apply base transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # EXTRA AUGMENTATION FOR NORMAL CLASS ONLY (Strategy 3)
        if label == 0 and self.extra_normal_aug:  # Normal class
            # Define extra aggressive augmentation for Normal
            extra_aug = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.8),
                A.HueSaturationValue(hue_shift_limit=25, sat_shift_limit=40, val_shift_limit=25, p=0.7),
                A.GaussianBlur(blur_limit=(3, 7), p=0.6),
                A.GaussNoise(var_limit=(20.0, 60.0), p=0.5),
                A.RandomGamma(gamma_limit=(70, 130), p=0.6),
                A.CLAHE(clip_limit=6.0, p=0.6),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.4),
            ])
            
            # Convert tensor back to numpy for extra augmentation
            if isinstance(image, torch.Tensor):
                image_np = image.permute(1, 2, 0).numpy()
                # Denormalize
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image_np = (image_np * std + mean) * 255.0
                image_np = np.clip(image_np, 0, 255).astype(np.uint8)
                
                # Apply extra augmentation
                extra_transformed = extra_aug(image=image_np)
                image_np = extra_transformed['image']
                
                # Re-normalize and convert back to tensor
                image_np = image_np.astype(np.float32) / 255.0
                image_np = (image_np - mean) / std
                image = torch.from_numpy(image_np).permute(2, 0, 1).float()
        
        return image, label


def get_transforms(train=True):
    if train:
        return A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.7),

            # Stronger augmentation to learn Normal variations better
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.8
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.6
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
            A.CLAHE(clip_limit=4.0, p=0.5),
            A.Blur(blur_limit=3, p=0.3),

            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])
