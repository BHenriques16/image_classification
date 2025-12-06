import torch
import torch.nn as nn
from torchvision import models

# Loads a pretrained model
def get_model(model_name, num_classes, feature_extract=True):
    model = None
    
    # ResNet18
    if model_name == "resnet18":
        model = models.resnet18(weights='DEFAULT')
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    # Mobilenet V2 (light and fast) 
    elif model_name == "mobilenet":
        model = models.mobilenet_v2(weights='DEFAULT')
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    # Efficientnet (balanced efficiency)
    elif model_name == "efficientnet":
        model = models.efficientnet_b0(weights='DEFAULT')
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    # Densenet 121 (Dense connections, good feature reuse)
    elif model_name == "densenet121":
        model = models.densenet121(weights='DEFAULT')
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)

    else:
        raise ValueError(f"Model {model_name} not implemented.")

    return model

def get_gradcam_target_layer(model, model_name):
    if model_name == "resnet18":
        return [model.layer4[-1]]
    elif model_name == "mobilenet":
        return [model.features[-1]]
    elif model_name == "efficientnet":
        return [model.features[-1]]
    elif model_name == "densenet121":
        return [model.features[-1]]
    else:
        raise ValueError(f"Unknown model for GradCAM: {model_name}")