import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm


# classes are no_pain (0) and pain (1)

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

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

# load pretrained VGG-19 model
# and use GPU if available
CUDA_VISIBLE_DEVICES=0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).to(device)
for param in vgg19.parameters():
    param.requires_grad = False  # freeze layers

vgg19.classifier[6] = nn.Linear(4096, 2).to(device)  # modify final layer for binary classification

# define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vgg19.classifier[6].parameters(), lr=0.001)

# training loop
num_epochs = 5
for epoch in range(num_epochs):
    vgg19.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = vgg19(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        progress_bar.set_postfix(loss=running_loss/len(train_loader))
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

# Save the model after each epoch
model_save_path = os.path.join(base_dir, '../model', f'vgg19_v0.pth')
torch.save(vgg19.state_dict(), model_save_path)

# TODO validation phase
