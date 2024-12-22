import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score, roc_auc_score
import random
import numpy as np
import time

# Fix random seed for reproducibility
def set_random_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Define image transformations
def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# Load datasets
def load_data(base_dir, transform):
    train_dir = os.path.join(base_dir, '../data/train-aug') # change this between train and train-aug
    val_dir = os.path.join(base_dir, '../data/val')
    train_data = datasets.ImageFolder(root=train_dir, transform=transform)
    val_data = datasets.ImageFolder(root=val_dir, transform=transform)
    return train_data, val_data

# Create data loaders
def get_data_loaders(train_data, val_data, batch_size=32):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

# Initialize model
def initialize_model(device):
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device)
    # Unfreeze the last few layers
    for param in list(resnet.parameters())[:-10]:
        param.requires_grad = False
    resnet.fc = nn.Linear(resnet.fc.in_features, 2).to(device)  # modify final layer for binary classification
    return resnet

# Train model
def train(model, loader, optimizer, criterion, device, accumulation_steps):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()

    for i, (inputs, labels) in enumerate(tqdm(loader, desc="Training")):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels) / accumulation_steps  # Normalize loss
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * accumulation_steps

    return running_loss / len(loader)

# Validate model
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = correct / total * 100
    f1 = f1_score(all_labels, all_preds, average='weighted')
    roc = roc_auc_score(all_labels, all_preds)
    return running_loss / len(loader), accuracy, f1, roc

# Hyperparameter tuning
def hyperparameter_tuning(base_dir, train_loader, val_loader, param_grid, device, save_path):
    best_params = None
    best_val_loss = float("inf")

    for params in ParameterGrid(param_grid):
        print(f"Testing with parameters: {params}")

        model = initialize_model(device)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=params['learning_rate'], weight_decay=params['weight_decay'])
        criterion = nn.CrossEntropyLoss()

        for epoch in range(params['num_epochs']):
            print(f"Epoch {epoch + 1}/{params['num_epochs']}")

            train_loss = train(model, train_loader, optimizer, criterion, device, params['accumulation_steps'])
            print(f"Training Loss: {train_loss:.4f}")

            val_loss, val_accuracy, val_f1, val_roc = validate(model, val_loader, criterion, device)
            print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%, F1 Score: {val_f1:.4f}, ROC AUC: {val_roc:.4f}")

            # Append validation loss, accuracy, and parameters to a text file
            # log_path = os.path.join(base_dir, '../model', 'cnn_training_log.txt')
            # with open(log_path, 'a') as log_file:
            #     log_file.write(f"Epoch: {epoch + 1}, Params: {params}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%\n")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = params
                print(f"New Best Params: {best_params}")
                model_save_path = os.path.join(base_dir, '../model', save_path)
                torch.save(model.state_dict(), model_save_path)
                print("Saved Best Model!")

    return best_params

# Main function
def main():
    set_random_seed()
    transform = get_transforms()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_data, val_data = load_data(base_dir, transform)
    train_loader, val_loader = get_data_loaders(train_data, val_data)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define hyperparameter grid
    param_grid = {
        'learning_rate': [0.0001],
        'weight_decay': [0.1],
        'num_epochs': [30],
        'accumulation_steps': [4]
    }

    # Perform hyperparameter tuning
    best_params = hyperparameter_tuning(base_dir, train_loader, val_loader, param_grid, device, f'resnet_besttfinal_model.pth')
    print(f"Best hyperparameters: {best_params}")

    # Save the best hyperparameters to a text file
    # best_params_path = os.path.join(base_dir, '../model', 'best_params_cnn.txt')
    # with open(best_params_path, 'w') as f:
    #     for key, value in best_params.items():
    #         f.write(f"{key}: {value}\n")
    # print(f"Best hyperparameters saved to {best_params_path}")

if __name__ == "__main__":
    main()