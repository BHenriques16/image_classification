import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch.nn.functional as F
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from data_preparation import create_data_transforms
from cnn import CNNModel

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

# Full training function with early stopping
def train(model, train_loader, val_loader, optimizer, criterion, max_epochs, device, model_save_path, patience=5):
    best_loss = float('inf')
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    early_stopping = EarlyStopping(patience=patience)

    total_start_time = time.time()

    for epoch in range(max_epochs):
        epoch_start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, epoch, max_epochs, device)
        val_loss, val_acc = validate(model, val_loader, criterion, val_loader.dataset, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{max_epochs} | Train Acc: {train_acc:.2f}% | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}% | Val Loss: {val_loss:.4f} | Epoch Time: {epoch_time:.2f} seconds")

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Best model saved (Validation Loss: {best_loss:.4f})")

    total_time = time.time() - total_start_time
    print(f"Training complete! Total training time: {total_time:.2f} seconds")

    # Plot training and validation metrics
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
    plt.show()

# Test evaluation function
def test(model, test_loader, criterion, test_dataset, device):
    model.eval()
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

    return test_loss / len(test_dataset), 100. * correct / total


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

    # Print dataset sizes for each partition
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Create Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Print class names
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    # Create model
    model = CNNModel(img_size, num_classes)
    model.to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train model with Early Stopping
    train(model, train_loader, val_loader, optimizer, criterion, epochs, device, model_save_path, patience=5)

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(model_save_path))
    test_loss, test_acc = test(model, test_loader, criterion, test_dataset, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")