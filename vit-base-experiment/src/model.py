import torch
import torch.nn as nn
import torchvision.models as models
import timm


def get_model(num_classes=2, pretrained=True, model_name='densenet121'):
    """
    Get model architecture
    
    Args:
        num_classes: Number of output classes
        pretrained: Use pretrained weights
        model_name: Model architecture to use
            Options: 'densenet121', 'vit_base', 'vit_small', 'vit_large', 
                    'swin_base', 'swin_small'
    """
    
    if model_name == 'densenet121':
        # Original DenseNet121
        model = models.densenet121(pretrained=pretrained)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, num_classes)
        
    elif model_name == 'vit_base':
        # Vision Transformer Base (86M parameters)
        model = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=num_classes)
        
    elif model_name == 'vit_small':
        # Vision Transformer Small (22M parameters) - Faster training
        model = timm.create_model('vit_small_patch16_224', pretrained=pretrained, num_classes=num_classes)
        
    elif model_name == 'vit_large':
        # Vision Transformer Large (304M parameters) - Best accuracy but slow
        model = timm.create_model('vit_large_patch16_224', pretrained=pretrained, num_classes=num_classes)
        
    elif model_name == 'swin_base':
        # Swin Transformer Base (88M parameters) - Good for medical images
        model = timm.create_model('swin_base_patch4_window7_224', pretrained=pretrained, num_classes=num_classes)
        
    elif model_name == 'swin_small':
        # Swin Transformer Small (50M parameters) - Balance speed/accuracy
        model = timm.create_model('swin_small_patch4_window7_224', pretrained=pretrained, num_classes=num_classes)
        
    elif model_name == 'vit_base_384':
        # ViT trained on 384x384 images (better for detail)
        model = timm.create_model('vit_base_patch16_384', pretrained=pretrained, num_classes=num_classes)

    elif model_name == 'swin_base_384':
        # Swin Transformer Base 384x384 (Better detail)
        model = timm.create_model('swin_base_patch4_window12_384', pretrained=pretrained, num_classes=num_classes)

    elif model_name == 'swin_large_patch4_window12_384':
        # Swin Transformer Large 384x384
        model = timm.create_model('swin_large_patch4_window12_384', pretrained=pretrained, num_classes=num_classes)

    elif model_name == 'vit_base_patch16_384':
        # ViT Base 384x384
        model = timm.create_model('vit_base_patch16_384', pretrained=pretrained, num_classes=num_classes)
        
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    print(f"✅ Loaded model: {model_name}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    return model.cuda() if torch.cuda.is_available() else model
