import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score


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



def evaluate_gender_specific(model, device, base_dir, ids, gender):
    criterion = nn.CrossEntropyLoss()  # define the loss function

    test_dir = os.path.join(base_dir, '../data/test')
    test_data = datasets.ImageFolder(root=test_dir, transform=transform)
    
    # filter the dataset to only include the specified ids
    indices = [i for i, (path, _) in enumerate(test_data.samples) if any(id in path for id in ids)]
    test_data = torch.utils.data.Subset(test_data, indices)
    
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
        for inputs, labels in tqdm(test_loader, desc=f"Evaluating {gender}", leave=False):
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

    return test_accuracy, f1, auc



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

    male_ids = [
        "048-aa048",
        "052-dr052",
        "064-ak064",
        "095-tv095",
        "096-bg096",
        "097-gf097",
        "101-mg101",
        "103-jk103",
        "109-ib109",
        "115-jy115",
        "120-kz120",
        "123-jh123"
    ]
    male_ids = [id.split('-')[1] for id in male_ids]  # extract the id from the full id, after the -

    female_ids = [
        "042-ll042",
        "043-jh043",
        "047-jl047",
        "049-bm049",
        "059-fn059",
        "066-mg066",
        "080-bn080",
        "092-ch092",
        "106-nm106",
        "107-hs107",
        "108-th108",
        "121-vw121",
        "124-dn124"
    ]
    female_ids = [id.split('-')[1] for id in female_ids]  # extract the id from the full id, after the -

    # # evaluate for male
    # accuracy_M, f1_M, roc_auc_M = evaluate_gender_specific(vgg19, device, base_dir, male_ids, "Male")

    # # evaluate for female
    # accuracy_F, f1_F, roc_auc_F = evaluate_gender_specific(vgg19, device, base_dir, female_ids, "Female")

    # print results
    # print("Accuracy Male:", accuracy_M)
    # print("Accuracy Female:", accuracy_F)
    # print("F1 score Male:", f1_M)
    # print("F1 score Female:", f1_F)
    # print("ROC AUC score Male:", roc_auc_M)
    # print("ROC AUC score Female:", roc_auc_F)

    evaluate_model(vgg19, device, base_dir)