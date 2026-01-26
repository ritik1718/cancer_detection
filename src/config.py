import torch


class Config:
    DATA_DIR = 'data/raw'
    TRAIN_DIR = 'data/raw/train'
    VAL_DIR = 'data/raw/val'
    TEST_DIR = 'data/raw/test'
    
    NUM_CLASSES = 2
    MODEL_NAME = 'densenet121'
    PRETRAINED = True
    
    BATCH_SIZE = 16
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 1e-4
    
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    MODEL_SAVE_PATH = 'models/saved_models'
    BEST_MODEL_PATH = 'models/saved_models/model_visionTransformer.pth'
    
    CLASS_NAMES = ['Normal', 'OSCC']
    
    PATIENCE = 10
    NUM_WORKERS = 0
