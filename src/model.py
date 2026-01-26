import torch
import torch.nn as nn
from torchvision import models
from src.config import Config

class DenseNet121Model(nn.Module):
    def __init__(self, num_classes=Config.NUM_CLASSES, pretrained=True):
        super(DenseNet121Model, self).__init__()
        
        if pretrained:
            self.model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        else:
            self.model = models.densenet121(weights=None)
        
        num_features = self.model.classifier.in_features
        
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

def get_model(num_classes=Config.NUM_CLASSES, pretrained=True):
    model = DenseNet121Model(num_classes=num_classes, pretrained=pretrained)
    model = model.to(Config.DEVICE)
    return model
