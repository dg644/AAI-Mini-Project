import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm


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

    with torch.no_grad():  # disable gradient computation
        for inputs, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss /= len(test_loader)
    test_accuracy = 100 * correct / total

    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
    print(f'Total: {total}, Correct: {correct}')
    return test_accuracy # might use ROC instead of accuracy

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).to(device)

    # modify the classifier layer to match the checkpoint's shape
    vgg19.classifier[6] = torch.nn.Linear(4096, 2)
    vgg19.load_state_dict(torch.load(os.path.join(base_dir, '../model/vgg19_v0.pth')))  # change the path to model
    vgg19.to(device)

    # define transform to be same as training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # evaluate the model
    evaluate_model(vgg19, device, base_dir)