import torch


class Config:
    DATA_DIR = '../data/raw'
    TRAIN_DIR = '../data/raw/train'
    VAL_DIR = '../data/raw/val'
    TEST_DIR = '../data/raw/test'
    
    NUM_CLASSES = 2
    
    # --- MODEL SELECTION ---
    # Options: 'densenet121', 'vit_base', 'vit_small', 'vit_large', 
    #          'swin_base', 'swin_small', 'vit_base_384', 'swin_base_384'
    MODEL_NAME = 'swin_base'
    PRETRAINED = True
    
    # Training hyperparameters
    BATCH_SIZE = 16  # Increased for 224
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    BEST_MODEL_PATH = 'models/best_swin_model.pth'
    
    CLASS_NAMES = ['Normal', 'OSCC']
    
    PATIENCE = 10
    NUM_WORKERS = 0
