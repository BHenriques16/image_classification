import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from data_transformation import create_data_transforms
from cnn import CNNModel
import numpy as np
import random
import os
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib.pyplot as plt

# Fix seed for reproducibility
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

# Early stopping class
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Training loop
def train_one_epoch(model, train_loader, optimizer, criterion, epoch, max_epochs, device):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{max_epochs}]")
    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()   
        outputs = model(inputs)
        loss = criterion(outputs, labels)   
        loss.backward()     
        optimizer.step()    

        train_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(loss=train_loss / total)
    return train_loss / total, 100. * correct / total

# Validation loop
def validate(model, val_loader, criterion, val_dataset, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    return val_loss / len(val_dataset), 100. * correct / total

# Full training function with early stopping and scheduler
def train(model, train_loader, val_loader, optimizer, criterion, max_epochs, device, model_save_path, patience=5):
    import datetime
    best_loss = float('inf')
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    early_stopping = EarlyStopping(patience=patience)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    total_start_time = time.time()

    for epoch in range(max_epochs):
        epoch_start_time = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, epoch, max_epochs, device)
        val_loss, val_acc = validate(model, val_loader, criterion, val_loader.dataset, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        scheduler.step(val_loss)
        epoch_time = time.time() - epoch_start_time
        epoch_end_dt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"Epoch {epoch+1}/{max_epochs} | Train Acc: {train_acc:.2f}% | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}% | Val Loss: {val_loss:.4f} | Epoch Time: {epoch_time:.2f} secs | Finished at: {epoch_end_dt}")

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Best model saved (Validation Loss: {best_loss:.4f})")

    total_time = time.time() - total_start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training complete! Total training time: {total_time:.2f} seconds ({total_time_str})")

    # Save and show plots
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.legend()
    plt.title("Accuracy")
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("Loss")
    plot_path = "results/train_metrics.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.show()
    print(f"Training plots saved to {plot_path}")

# MAIN
if __name__ == "__main__":

    # Hiperparameters
    train_dir = "dataset_waste_container"
    batch_size = 16
    img_size = 224
    epochs = 50
    lr = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_save_path = "models/best_model.pth"

    print('Using device:', device)

    # Create data transforms
    train_transforms, val_transforms = create_data_transforms(img_size)

    # Load the dataset with train transforms temporarily
    full_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)

    # Split dataset into train/val/test (70/15/15)
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    # Assign val transforms to val and test datasets
    val_dataset.dataset.transform = val_transforms
    test_dataset.dataset.transform = val_transforms

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Create Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    # Create model
    model = CNNModel(img_size, num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train(model, train_loader, val_loader, optimizer, criterion, epochs, device, model_save_path, patience=5)