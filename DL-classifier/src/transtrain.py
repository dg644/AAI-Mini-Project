import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms
from torch.cuda.amp import GradScaler, autocast
import timm  # Install timm: pip install timm
from tqdm import tqdm
import os


# hyperparameters
num_classes = 2  # pain or no_pain
batch_size = 16
image_size = 224  # For ViT
learning_rate = 1e-4
num_epochs = 20
accumulation_steps = 4  # simulate larger batch sizes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# load dataset
base_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(base_dir, '../data/train')
val_dir = os.path.join(base_dir, '../data/val')
test_dir = os.path.join(base_dir, '../data/test')
train_data = datasets.ImageFolder(root=train_dir, transform=transform)
val_data = datasets.ImageFolder(root=val_dir, transform=transform)
test_data = datasets.ImageFolder(root=test_dir, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# load a pretrained Vision Transformer model
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
model.to(device)

# loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# mixed precision scaler
scaler = GradScaler()

def training_step(model, loader, optimizer, criterion, scaler, accumulation_steps):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()

    for i, (images, labels) in enumerate(tqdm(loader, desc="Training")):
        images, labels = images.to(device), labels.to(device)

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels) / accumulation_steps  # Normalize loss

        scaler.scale(loss).backward()
        
        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * accumulation_steps

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


best_val_loss = float("inf")
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    # training
    train_loss = training_step(model, train_loader, optimizer, criterion, scaler, accumulation_steps)
    print(f"Training Loss: {train_loss:.4f}")

    # validation
    val_loss, val_accuracy = validate(model, val_loader, criterion)
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "vit_best_model.pth")
        print("Saved Best Model!")


# print("Testing the best model...")
# model.load_state_dict(torch.load("best_model.pth"))
# test_loss, test_accuracy = validate(model, test_loader, criterion)
# print(f"Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%")
