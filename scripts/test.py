import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
import os
from data_transformation import create_data_transforms
from cnn import CNNModel
from train_cnn import FocalLoss
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Load test indices (from split during training)
test_indices = np.load("results/test_indices.npy")

# Configuration
test_dir = "dataset_waste_container"
img_size = 224
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "models/best_model.pth"
N_EXAMPLES = 6

# Data Loading
_, test_transforms = create_data_transforms(img_size)
full_dataset = datasets.ImageFolder(root=test_dir, transform=test_transforms)
class_names = full_dataset.classes
num_classes = len(class_names)
test_dataset = Subset(full_dataset, test_indices)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Model load
model = CNNModel(img_size, num_classes)
model.to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Focal loss function
criterion = FocalLoss(alpha=1, gamma=2)

# Grad-CAM Visualization: (side by side)
try:
    shown_classes = set()
    shown_indices = []
    for idx in range(len(test_dataset)):
        img, label = test_dataset[idx]
        if label not in shown_classes:
            shown_classes.add(label)
            shown_indices.append(idx)
        if len(shown_indices) == N_EXAMPLES:
            break

    print(f"\nGrad-CAM visualization for {len(shown_indices)} test images:")
    fig, axes = plt.subplots(nrows=2, ncols=N_EXAMPLES, figsize=(4*N_EXAMPLES, 6))
    for i, idx in enumerate(shown_indices):
        img, label = test_dataset[idx]
        input_tensor = img.unsqueeze(0).to(device)
        target_layers = [model.conv4]
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=(device.type=='cuda'))
        targets = [ClassifierOutputTarget(label)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        img_np = img.permute(1,2,0).cpu().numpy()
        img_np = (img_np * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406])).clip(0,1)
        img_cam = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        # Original above
        axes[0, i].imshow(img_np)
        axes[0, i].set_title(f"Original\nTrue: {class_names[label]}")
        axes[0, i].axis('off')
        axes[1, i].imshow(img_cam)
        axes[1, i].set_title("Grad-CAM")
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.savefig("results/gradcam_test_examples_horizontal.png")
    plt.show()

except Exception as e:
    print("[Grad-CAM ERROR]", str(e))