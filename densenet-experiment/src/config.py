import torch
import os

class Config:
    # Using absolute paths to point to the existing data
    DATA_DIR = r'c:\Projects\oral-cancer-detection\data\raw'
    TRAIN_DIR = r'c:\Projects\oral-cancer-detection\data\raw\train'
    VAL_DIR = r'c:\Projects\oral-cancer-detection\data\raw\val'
    TEST_DIR = r'c:\Projects\oral-cancer-detection\data\raw\test'
    
    NUM_CLASSES = 2
    
    # --- MODEL SELECTION ---
    MODEL_NAME = 'densenet121' # Targeted for high accuracy
    PRETRAINED = True
    
    # Training hyperparameters
    BATCH_SIZE = 32 # Increased for DenseNet efficiency
    NUM_EPOCHS = 40 # Increased to allow for better convergence
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    
    IMG_SIZE = 224 # Standard for DenseNet
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    MODEL_SAVE_PATH = 'models/saved_models'
    BEST_MODEL_PATH = 'models/saved_models/best_model_densenet.pth'
    
    CLASS_NAMES = ['Normal', 'OSCC']
    
    PATIENCE = 10
    NUM_WORKERS = 0
