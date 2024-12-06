import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from sklearn.metrics import f1_score, roc_auc_score
import random
import numpy as np


# classes are no_pain (0) and pain (1)

# define image transformations
# fix random seed for reproducibility
random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomHorizontalFlip(p=0.5),
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
    for i, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = vgg19(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        progress_bar.set_postfix(loss=running_loss/len(train_loader))
        
        # Add a break every 100 batches to let hardware cool down
        if i % 100 == 0 and i > 0:
            print("\nTaking a short break to cool down...")
            torch.cuda.empty_cache()
            time.sleep(10)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

    # validation phase
    vgg19.eval()  # set model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():  # disable gradient computation
        progress_bar = tqdm(val_loader, desc="Validation")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = vgg19(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            progress_bar.set_postfix(loss=val_loss/len(val_loader), accuracy=100 * correct / total)

    val_loss /= len(val_loader)
    accuracy = 100 * correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted')
    roc = roc_auc_score(all_labels, all_preds)
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.2f}%')
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation F1 Score: {f1:.4f}, Validation ROC AUC: {roc:.4f}')

# save the model (put in loop to save after each epoch)
model_save_path = os.path.join(base_dir, '../model', f'vgg19_v2.pth')
torch.save(vgg19.state_dict(), model_save_path)

# # validation phase
# vgg19.eval()  # set model to evaluation mode
# val_loss = 0.0
# correct = 0
# total = 0

# with torch.no_grad():  # disable gradient computation
#     for inputs, labels in val_loader:
#         inputs, labels = inputs.to(device), labels.to(device)
#         outputs = vgg19(inputs)
#         loss = criterion(outputs, labels)
#         val_loss += loss.item()
        
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# val_loss /= len(val_loader)
# accuracy = 100 * correct / total

# print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.2f}%')