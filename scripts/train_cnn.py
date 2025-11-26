import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from data_transformation import create_data_transforms
from cnn import CNNModel

# Focal Loss Implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Seed for reproducibility
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

# Data splitting (stratified) and indices saving
train_dir = "dataset_waste_container"
batch_size = 16
img_size = 224

train_transforms, val_transforms = create_data_transforms(img_size)
full_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
targets = np.array([sample[1] for sample in full_dataset.samples])

# Stratified train/val/test split (70/15/15)
test_size = 0.15
val_size = 0.15

sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
for train_val_idx, test_idx in sss1.split(np.zeros(len(targets)), targets):
    pass

train_val_targets = targets[train_val_idx]
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size/(1-test_size), random_state=SEED)
for train_idx, val_idx in sss2.split(np.zeros(len(train_val_targets)), train_val_targets):
    pass

train_indices = train_val_idx[train_idx]
val_indices = train_val_idx[val_idx]

# save the indices
os.makedirs("results", exist_ok=True)
np.save("results/train_indices.npy", train_indices)
np.save("results/val_indices.npy", val_indices)
np.save("results/test_indices.npy", test_idx)

print("Splits salvos em 'results/'.")

# create the datasets using subset
train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)
test_dataset = Subset(full_dataset, test_idx)

full_dataset.transform = val_transforms  # only affects val/test

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Class information
class_names = full_dataset.classes
num_classes = len(class_names)
print(f"Found {num_classes} classes: {class_names}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel(img_size, num_classes).to(device)
criterion = FocalLoss(alpha=1, gamma=2)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Early Stopping
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
        print(f"Epoch {epoch+1}/{max_epochs} | Train Acc: {train_acc:.2f}% | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}% | Val Loss: {val_loss:.4f} | Epoch Time: {epoch_time:.2f} secs")
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Best model saved (Validation Loss: {best_loss:.4f})")
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
    total_time = time.time() - total_start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training complete! Total training time: {total_time:.2f} seconds ({total_time_str})")

# MAIN
if __name__ == "__main__":
    print('Using device:', device)
    model_save_path = "models/best_model.pth"
    epochs = 50
    train(model, train_loader, val_loader, optimizer, criterion, epochs, device, model_save_path, patience=5)