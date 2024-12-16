import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import timm

def load_cnn_model(device):
    vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).to(device)
    vgg19.classifier[6] = torch.nn.Linear(4096, 2)
    vgg19.load_state_dict(torch.load(os.path.join(base_dir, '../model/vgg19_best_model.pth')))
    return vgg19

# Replace this with the actual loading code for your transformer model
def load_transformer_model(device):
    num_classes = 2
    transformer_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes).to(device)
    transformer_model.load_state_dict(torch.load(os.path.join(base_dir, '../model/vit_best_model.pth')))
    return transformer_model

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

    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():  # disable gradient computation
        for inputs, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # store labels and predicted probabilities
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    test_loss /= len(test_loader)
    test_accuracy = 100 * correct / total

    auc = roc_auc_score(all_labels, all_probs)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
    print(f'Total: {total}, Correct: {correct}')
    print(f'F1 Score: {f1:.4f}, ROC AUC: {auc:.4f}')

    return test_accuracy, f1, auc

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define transform to be same as training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model_type = 'cnn'  # Change this to 'transformer' to evaluate the transformer model

    if model_type == 'cnn':
        model = load_cnn_model(device)
    elif model_type == 'transformer':
        model = load_transformer_model(device)
    else:
        raise ValueError("Invalid model type. Choose 'cnn' or 'transformer'.")

    evaluate_model(model, device, base_dir)