import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import os
from data_transformation import create_data_transforms
from models_factory import get_model, get_gradcam_target_layer 
from train_cnn import FocalLoss
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Configurations
MODEL_NAME = "densenet121"   # Options: "resnet18", "mobilenet", "efficientnet", "densenet121"

test_indices = np.load("results/test_indices.npy")
test_dir = "dataset_waste_container"
img_size = 224
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = f"models/{MODEL_NAME}_best.pth"

print(f"testing model: {MODEL_NAME} loaded from {model_path}")

# Data
_, test_transforms = create_data_transforms(img_size)
full_dataset = datasets.ImageFolder(root=test_dir, transform=test_transforms)
class_names = full_dataset.classes
num_classes = len(class_names)
test_dataset = Subset(full_dataset, test_indices)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load Model
model = get_model(MODEL_NAME, num_classes).to(device)
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
else:
    print(f"ERROR")
    exit()

model.eval()

# Evaluation loop
all_preds = []
all_labels = []
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

acc = 100. * correct / total
print(f"Accuracy Final ({MODEL_NAME}): {acc:.2f}%")
print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

# Grad-cam
try:
    shown_classes = set()
    shown_indices = []
    for idx in range(len(test_dataset)):
        _, label = test_dataset[idx]
        if label not in shown_classes:
            shown_classes.add(label)
            shown_indices.append(idx)
        if len(shown_indices) == 7: break

    target_layers = get_gradcam_target_layer(model, MODEL_NAME)
    
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=(device.type=='cuda'))

    fig, axes = plt.subplots(2, 7, figsize=(20, 6))
    for i, idx in enumerate(shown_indices):
        img, label = test_dataset[idx]
        input_tensor = img.unsqueeze(0).to(device)
        
        targets = [ClassifierOutputTarget(label)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        
        img_np = img.permute(1,2,0).cpu().numpy()
        img_np = (img_np * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406])).clip(0,1)
        img_cam = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        
        axes[0, i].imshow(img_np)
        axes[0, i].set_title(f"{class_names[label]}")
        axes[0, i].axis('off')
        axes[1, i].imshow(img_cam)
        axes[1, i].axis('off')
        
    plt.suptitle(f"Grad-CAM: {MODEL_NAME}", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"results/grad_cam_{MODEL_NAME}.png")
    plt.show()

except Exception as e:
    print(f"Erro no GradCAM: {e}")