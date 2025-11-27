import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import datetime
from data_transformation import create_data_transforms
from models_factory import get_model 
from train_cnn import FocalLoss, EarlyStopping, train_one_epoch, validate 

# Configuration
MODEL_NAME = "densenet121"  # Options: "resnet18", "mobilenet", "efficientnet", "densenet121"
BATCH_SIZE = 16          
IMG_SIZE = 224
EPOCHS = 50
LEARNING_RATE = 0.0001

def main():
    # Set seed and device
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training model: {MODEL_NAME} on {device}")

    # Load Indices
    if not os.path.exists("results/train_indices.npy"):
        raise FileNotFoundError("Run the original script first to generate splits (train_indices.npy)!")

    train_indices = np.load("results/train_indices.npy")
    val_indices = np.load("results/val_indices.npy")

    # Dataset and loaders
    train_dir = "dataset_waste_container"
    train_transforms, val_transforms = create_data_transforms(IMG_SIZE)
    
    # Load the full dataset just to get class names and number of classes
    full_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    class_names = full_dataset.classes
    num_classes = len(class_names)

    # Create separate datasets to ensure correct transforms (augmentation vs clean)
    train_ds_clean = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_ds_clean = datasets.ImageFolder(train_dir, transform=val_transforms)
    
    train_subset = Subset(train_ds_clean, train_indices)
    val_subset = Subset(val_ds_clean, val_indices)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Initialize model 
    model = get_model(MODEL_NAME, num_classes).to(device)

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = FocalLoss(alpha=1, gamma=2)

    # Training Loop
    save_path = f"models/{MODEL_NAME}_best.pth"
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    best_loss = float('inf')
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    early_stopping = EarlyStopping(patience=5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    start_time = time.time()

    for epoch in range(EPOCHS):
        t_loss, t_acc = train_one_epoch(model, train_loader, optimizer, criterion, epoch, EPOCHS, device)
        v_loss, v_acc = validate(model, val_loader, criterion, val_subset, device) 
        
        train_losses.append(t_loss)
        val_losses.append(v_loss)
        train_accs.append(t_acc)
        val_accs.append(v_acc)
        
        scheduler.step(v_loss)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Acc: {t_acc:.2f}% | Val Acc: {v_acc:.2f}% | Val Loss: {v_loss:.4f}")
        
        early_stopping(v_loss)
        if v_loss < best_loss:
            best_loss = v_loss
            torch.save(model.state_dict(), save_path)
            print(f"--> Best model saved: {save_path}")
            
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # Plots
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_accs, label="Train")
    plt.plot(val_accs, label="Val")
    plt.title(f"Accuracy - {MODEL_NAME}")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.title(f"Loss - {MODEL_NAME}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/metrics_{MODEL_NAME}.png")
    print(f"Training complete. Plots saved to results/metrics_{MODEL_NAME}.png")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()