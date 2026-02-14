import torch


class Config:
    DATA_DIR = r'c:\Projects\oral-cancer-detection\data\raw'
    TRAIN_DIR = r'c:\Projects\oral-cancer-detection\data\raw\train'
    VAL_DIR = r'c:\Projects\oral-cancer-detection\data\raw\val'
    TEST_DIR = r'c:\Projects\oral-cancer-detection\data\raw\test'
    
    NUM_CLASSES = 2
    
    # --- MODEL SELECTION ---
    # Options: 'densenet121', 'vit_base', 'vit_small', 'vit_large', 
    #          'swin_base', 'swin_small', 'vit_base_384', 'swin_base_384'
    MODEL_NAME = 'swin_large_patch4_window12_384'
    PRETRAINED = True
    
    # Training hyperparameters
    BATCH_SIZE = 4  # Reduced for 384 Large
    NUM_EPOCHS = 30
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 1e-4
    
    IMG_SIZE = 384
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    BEST_MODEL_PATH = 'models/best_swin_model_large.pth'
    
    CLASS_NAMES = ['Normal', 'OSCC']
    
    PATIENCE = 10
    NUM_WORKERS = 0
