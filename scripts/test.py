import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import os
from data_transformation import create_data_transforms
from cnn import CNNModel 

# Configurations
test_dir = "dataset_waste_container"
img_size = 224
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "models/best_model.pth"
N_EXAMPLES = 5

# Data Loading
_, test_transforms = create_data_transforms(img_size)
test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
class_names = test_dataset.classes
num_classes = len(class_names)

# Model load
model = CNNModel(img_size, num_classes)
model.to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

criterion = torch.nn.CrossEntropyLoss()

# Test load & evaluation
all_preds = []
all_labels = []
test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        test_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_loss = test_loss / len(test_dataset)
test_acc = 100. * correct / total

labels_present = sorted(set(all_labels))
class_names_present = [class_names[l] for l in labels_present]

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
print("\nClassification report (classes presentes):")
class_report = classification_report(
    all_labels, all_preds,
    target_names=class_names_present,
    labels=labels_present,
    digits=4,
    zero_division=0
)
print(class_report)
cmatrix = confusion_matrix(all_labels, all_preds, labels=labels_present)
print("Confusion matrix:")
print(cmatrix)

os.makedirs("results", exist_ok=True)
with open("results/classification_report_test.txt", "w") as f:
    f.write(class_report)
np.save("results/confusion_matrix_test.npy", cmatrix)
print("Classification report and confusion matrix saved in /results/")

# Grad-cam Visualization: Show diferent classes examples
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    # Select up to n examples of different classes of the test set
    shown_classes = set()
    shown_indices = []
    for idx in range(len(test_dataset)):
        img, label = test_dataset[idx]
        if label not in shown_classes:
            shown_classes.add(label)
            shown_indices.append(idx)
        if len(shown_indices) == N_EXAMPLES:
            break
    # If there are missing examples, complete with random examples
    while len(shown_indices) < N_EXAMPLES and len(shown_indices) < len(test_dataset):
        if len(shown_indices) not in shown_indices:
            shown_indices.append(len(shown_indices))
    
    print(f"\nGrad-CAM visualization for {len(shown_indices)} test images:")
    fig, axes = plt.subplots(len(shown_indices), 2, figsize=(8, 4 * len(shown_indices)))
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
        axes[i,0].imshow(img_np)
        axes[i,0].set_title(f"Original\nTrue: {class_names[label]}")
        axes[i,0].axis('off')
        axes[i,1].imshow(img_cam)
        axes[i,1].set_title("Grad-CAM")
        axes[i,1].axis('off')
    plt.tight_layout()
    plt.savefig("results/gradcam_test_examples.png")
    plt.show()
    print(f"Grad-CAM images saved to /results/gradcam_test_examples.png")

except Exception as e:
    print("[Grad-CAM ERROR]", str(e))