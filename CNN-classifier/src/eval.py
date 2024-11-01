import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt


def evaluate_model(model, device, base_dir):
    criterion = nn.CrossEntropyLoss()  # define the loss function
        
    test_dir = os.path.join(base_dir, '../data/test')
    test_data = datasets.ImageFolder(root=test_dir, transform=transform)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    model.to(device)
    model.eval()  # set model to evaluation mode
    test_loss = 0.0
    correct = 0
    total = 0

    all_losses = []
    all_accuracies = []

    with torch.no_grad():  # disable gradient computation
        for inputs, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_losses.append(loss.item())
            all_accuracies.append(100 * (predicted == labels).sum().item() / labels.size(0))

    test_loss /= len(test_loader)
    test_accuracy = 100 * correct / total

    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
    print(f'Total: {total}, Correct: {correct}')

    # Plot the loss
    plt.figure()
    plt.plot(all_losses, color='blue', lw=2, label='Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Test Loss per Batch')
    plt.legend(loc="upper right")
    plt.show()

    # Plot the accuracy
    plt.figure()
    plt.plot(all_accuracies, color='green', lw=2, label='Accuracy')
    plt.xlabel('Batch')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy per Batch')
    plt.legend(loc="lower right")
    plt.show()

    return test_accuracy



def evaluate_model_with_roc(model, device, base_dir):
    criterion = nn.CrossEntropyLoss()  # define the loss function
        
    test_dir = os.path.join(base_dir, '../data/test')
    test_data = datasets.ImageFolder(root=test_dir, transform=transform)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    model.to(device)
    model.eval()  # set model to evaluation mode
    test_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_probs = []

    with torch.no_grad():  # disable gradient computation
        for inputs, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Store labels and predicted probabilities
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())

    test_loss /= len(test_loader)
    test_accuracy = 100 * correct / total

    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
    print(f'Total: {total}, Correct: {correct}')

    # Calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)

    # Calculate the AUC (Area Under the Curve)
    auc = roc_auc_score(all_labels, all_probs)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    return test_accuracy, auc



if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).to(device)

    # modify the classifier layer to match the checkpoint's shape
    vgg19.classifier[6] = torch.nn.Linear(4096, 2)
    vgg19.load_state_dict(torch.load(os.path.join(base_dir, '../model/vgg19_v1.pth')))  # change the path to model
    vgg19.to(device)

    # define transform to be same as training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # evaluate the model
    evaluate_model_with_roc(vgg19, device, base_dir)