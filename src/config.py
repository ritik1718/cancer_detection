import torch


class Config:
    DATA_DIR = 'data/raw'
    TRAIN_DIR = 'data/raw/train'
    VAL_DIR = 'data/raw/val'
    TEST_DIR = 'data/raw/test'
    
    NUM_CLASSES = 2
    
    # --- MODEL SELECTION ---
    # Options: 'densenet121', 'vit_base', 'vit_small', 'vit_large', 
    #          'swin_base', 'swin_small', 'vit_base_384', 'swin_base_384'
    MODEL_NAME = 'swin_base_384'  # ✅ Changed to higher res model
    PRETRAINED = True
    
    # Training hyperparameters
    BATCH_SIZE = 8  # Reduced for larger model
    NUM_EPOCHS = 30
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 1e-4
    
    IMG_SIZE = 384  # Updated for Swin-384
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    MODEL_SAVE_PATH = 'models/saved_models'
    BEST_MODEL_PATH = 'models/saved_models/best_model_swin_384.pth'  # New name to keep previous model
    
    CLASS_NAMES = ['Normal', 'OSCC']
    
    PATIENCE = 10
    NUM_WORKERS = 0
