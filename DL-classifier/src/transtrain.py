import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.amp import GradScaler, autocast
import timm
from tqdm import tqdm
import os
from sklearn.model_selection import ParameterGrid
import time


# hyperparameters
num_classes = 2  # pain or no_pain
batch_size = 16
image_size = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# load dataset
base_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(base_dir, '../data/train-aug') # change this between train and train-aug
val_dir = os.path.join(base_dir, '../data/val')
test_dir = os.path.join(base_dir, '../data/test')
train_data = datasets.ImageFolder(root=train_dir, transform=transform)
val_data = datasets.ImageFolder(root=val_dir, transform=transform)
test_data = datasets.ImageFolder(root=test_dir, transform=transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# load a pretrained Vision Transformer model
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
model.to(device)

# loss function and optimizer
criterion = nn.CrossEntropyLoss()

# mixed precision scaler
scaler = GradScaler()

def train(model, loader, optimizer, criterion, scaler, accumulation_steps):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()

    for i, (images, labels) in enumerate(tqdm(loader, desc="Training")):
        images, labels = images.to(device), labels.to(device)

        with autocast("cuda" if torch.cuda.is_available() else "cpu"):
            outputs = model(images)
            loss = criterion(outputs, labels) / accumulation_steps  # Normalize loss

        scaler.scale(loss).backward()
        
        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * accumulation_steps

        # sleep for 10 seconds to let laptop cool down
        if (i + 1) % (200) == 0:
            print("Sleeping for 10 seconds...")
            time.sleep(10)

    return running_loss / len(loader)


def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total * 100
    return running_loss / len(loader), accuracy

def hyperparameter_tuning(train_loader, val_loader, model, param_grid):
    best_params = None
    best_val_loss = float("inf")

    for params in ParameterGrid(param_grid):
        print(f"Testing with parameters: {params}")
        
        # Update model, optimizer, and other hyperparameters
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
        model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
        criterion = nn.CrossEntropyLoss()
        scaler = GradScaler()
        
        for epoch in range(params['num_epochs']):
            print(f"Epoch {epoch + 1}/{params['num_epochs']}")

            # training
            train_loss = train(model, train_loader, optimizer, criterion, scaler, params['accumulation_steps'])
            print(f"Training Loss: {train_loss:.4f}")

            # validation
            val_loss, val_accuracy = validate(model, val_loader, criterion)
            print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")

            best_params = params
            model_save_path = os.path.join(base_dir, '../model', f'vit_bestfinal_model.pth')
            torch.save(model.state_dict(), model_save_path)
            print("Saved Best Model!")

    return best_params

# Define hyperparameter grid
param_grid = {
    'learning_rate': [0.0001],
    'weight_decay': [0.1],
    'num_epochs': [10],
    'accumulation_steps': [2]
}

# Perform hyperparameter tuning
best_params = hyperparameter_tuning(train_loader, val_loader, model, param_grid)
print(f"Best hyperparameters: {best_params}")

best_params_path = os.path.join(base_dir, '../model', 'best_params_transformer.txt')
with open(best_params_path, 'w') as f:
    for key, value in best_params.items():
        f.write(f"{key}: {value}\n")
print(f"Best hyperparameters saved to {best_params_path}")

# Save the best hyperparameters to a text file

# print("Testing the best model...")
# model.load_state_dict(torch.load("best_model.pth"))
# test_loss, test_accuracy = validate(model, test_loader, criterion)
# print(f"Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%")
