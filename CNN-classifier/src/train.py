import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

base_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(base_dir, '../data/train')
val_dir = os.path.join(base_dir, '../data/val')
train_data = datasets.ImageFolder(root=train_dir, transform=transform)
val_data = datasets.ImageFolder(root=val_dir, transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# Load pretrained VGG-19 model
vgg19 = models.vgg19(pretrained=True)
for param in vgg19.parameters():
    param.requires_grad = False  # Freeze layers

vgg19.classifier[6] = nn.Linear(4096, 2)  # Modify final layer for binary classification

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg19.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vgg19.classifier[6].parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    vgg19.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = vgg19(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

    # Validation phase (optional)
    # Add validation logic here (similar to the code in the first answer)
