import torch
import torch.nn as nn
from torchvision import models

def get_model(model_name, num_classes, feature_extract=True):
    """
    Loads a pre-trained model and adjusts the final layer for the number of classes.
    """
    model = None
    
    # --- RESNET 18 ---
    if model_name == "resnet18":
        # 'DEFAULT' weights load the best available pre-training (ImageNet)
        model = models.resnet18(weights='DEFAULT')
        # Replace the final linear layer (fc)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    # --- MOBILENET V2 (light and fast) ---
    elif model_name == "mobilenet":
        model = models.mobilenet_v2(weights='DEFAULT')
        # In MobileNet the classification head is in 'classifier[1]'
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    # --- EFFICIENTNET B0 (balanced efficiency) ---
    elif model_name == "efficientnet":
        model = models.efficientnet_b0(weights='DEFAULT')
        # In EfficientNet the classification head is in 'classifier[1]'
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    # --- DENSENET 121 (Dense connections, good feature reuse) ---
    elif model_name == "densenet121":
        model = models.densenet121(weights='DEFAULT')
        # In DenseNet the classification layer is named 'classifier'
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)

    else:
        raise ValueError(f"Model {model_name} not implemented.")

    return model

def get_gradcam_target_layer(model, model_name):
    """
    Retrieves the target layer for Grad-CAM depending on the architecture.
    Usually focuses on the last convolutional feature map.
    """
    if model_name == "resnet18":
        return [model.layer4[-1]]
    elif model_name == "mobilenet":
        # Last convolutional feature layer
        return [model.features[-1]]
    elif model_name == "efficientnet":
        return [model.features[-1]]
    elif model_name == "densenet121":
        # The features in DenseNet are in a block named 'features'
        # The last layer inside features includes the final dense block and norm
        return [model.features[-1]]
    else:
        raise ValueError(f"Unknown model for GradCAM: {model_name}")