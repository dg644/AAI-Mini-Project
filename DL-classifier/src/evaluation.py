import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import timm
from sklearn.metrics import confusion_matrix

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
male_ids = [id.split('-')[1] for id in male_ids]     #extract the id from the full id, after the -

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
female_ids = [id.split('-')[1] for id in female_ids]    #extract the id from the full id, after the -

def load_cnn_model(device):
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device)
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, 2)
    resnet.load_state_dict(torch.load(os.path.join(base_dir, '../model/resnet_bestfinal_model.pth'), weights_only=True))
    return resnet

# Replace this with the actual loading code for your transformer model
def load_transformer_model(device):
    num_classes = 2
    transformer_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes).to(device)
    transformer_model.load_state_dict(torch.load(os.path.join(base_dir, '../model/vit_bestfinal_model.pth'), weights_only=True))
    return transformer_model

def calculate_fairness_metrics(labels, preds, ids):
    print(f"Length of labels: {len(labels)}")
    print(f"Length of preds: {len(preds)}")
    print(f"Length of ids: {len(ids)}")

    male_indices = [i for i, id in enumerate(ids) if id in male_ids and i < len(labels)]
    female_indices = [i for i, id in enumerate(ids) if id in female_ids and i < len(labels)]

    male_labels = [labels[i] for i in male_indices]
    male_preds = [preds[i] for i in male_indices]
    female_labels = [labels[i] for i in female_indices]
    female_preds = [preds[i] for i in female_indices]

    male_accuracy = sum([1 for i in range(len(male_labels)) if male_labels[i] == male_preds[i]]) / len(male_labels)
    female_accuracy = sum([1 for i in range(len(female_labels)) if female_labels[i] == female_preds[i]]) / len(female_labels)

    male_cm = confusion_matrix(male_labels, male_preds)
    female_cm = confusion_matrix(female_labels, female_preds)

    male_tpr = male_cm[1, 1] / (male_cm[1, 1] + male_cm[1, 0])
    female_tpr = female_cm[1, 1] / (female_cm[1, 1] + female_cm[1, 0])

    male_fpr = male_cm[0, 1] / (male_cm[0, 1] + male_cm[0, 0])
    female_fpr = female_cm[0, 1] / (female_cm[0, 1] + female_cm[0, 0])

    # Correcting the majority gender class
    majority_tpr = female_tpr
    minority_tpr = male_tpr
    majority_fpr = female_fpr
    minority_fpr = male_fpr
    majority_accuracy = female_accuracy
    minority_accuracy = male_accuracy

    # Confusion matrix details
    male_tp = male_cm[1, 1]
    male_fp = male_cm[0, 1]
    male_tn = male_cm[0, 0]
    male_fn = male_cm[1, 0]

    female_tp = female_cm[1, 1]
    female_fp = female_cm[0, 1]
    female_tn = female_cm[0, 0]
    female_fn = female_cm[1, 0]

    #fairness metrics

    equal_accuracy = abs(minority_accuracy - majority_accuracy)
    equal_opportunity = abs(minority_tpr - majority_tpr)
    equalized_odds = abs((minority_tpr - majority_tpr) + (minority_fpr - majority_fpr)) / 2

    male_pred_positive_rate = sum(male_preds) / len(male_preds)
    female_pred_positive_rate = sum(female_preds) / len(female_preds)
    print(f"Male pred positive rate: {male_pred_positive_rate:.4f}")
    print(f"Female pred positive rate: {female_pred_positive_rate:.4f}")
    disparate_impact = male_pred_positive_rate / female_pred_positive_rate

    # Additional fairness metrics
    demographic_parity = abs(male_pred_positive_rate - female_pred_positive_rate)
    treatment_equality = abs(male_fn / male_fp - female_fn / female_fp)

    #P(Y=pain|Y_hat=pain, Male)
    male_pos_pos_prob = sum([1 for i in range(len(male_labels)) if male_labels[i] == 1 and male_preds[i] == 1]) / len(male_labels)
    male_pos_pos_cond_prob = male_pos_pos_prob / male_pred_positive_rate
    #P(Y=pain|Y_hat=no-pain, Male)
    male_pos_neg_prob = sum([1 for i in range(len(male_labels)) if male_labels[i] == 1 and male_preds[i] == 0]) / len(male_labels)
    male_pos_neg_cond_prob = male_pos_neg_prob / (1 - male_pred_positive_rate)

    female_pos_pos_prob = sum([1 for i in range(len(female_labels)) if female_labels[i] == 1 and female_preds[i] == 1]) / len(female_labels)
    female_pos_pos_cond_prob = female_pos_pos_prob / female_pred_positive_rate
    female_pos_neg_prob = sum([1 for i in range(len(female_labels)) if female_labels[i] == 1 and female_preds[i] == 0]) / len(female_labels)
    female_pos_neg_cond_prob = female_pos_neg_prob / (1 - female_pred_positive_rate)

        #take the average difference for the two different predicted value cases
    test_fairness = abs((male_pos_pos_cond_prob - female_pos_pos_cond_prob) + (male_pos_neg_cond_prob - female_pos_neg_cond_prob)) / 2

    # male_positive_rate = sum([1 for i in range(len(male_labels)) if male_labels[i] == 1]) / len(male_labels)
    # female_positive_rate = sum([1 for i in range(len(female_labels)) if female_labels[i] == 1]) / len(female_labels)
    conditional_statistical_parity = 1

    # Confusion matrix details
    male_tp = male_cm[1, 1]
    male_fp = male_cm[0, 1]
    male_tn = male_cm[0, 0]
    male_fn = male_cm[1, 0]

    female_tp = female_cm[1, 1]
    female_fp = female_cm[0, 1]
    female_tn = female_cm[0, 0]
    female_fn = female_cm[1, 0]

    return equal_accuracy, equal_opportunity, equalized_odds, disparate_impact, demographic_parity, treatment_equality, test_fairness, conditional_statistical_parity, male_tp, male_fp, male_tn, male_fn, female_tp, female_fp, female_tn, female_fn

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
    all_ids = []

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
            batch_ids = [os.path.basename(path)[:5] for path, _ in test_loader.dataset.samples[total - labels.size(0):total]]
            all_ids.extend(batch_ids)

    test_loss /= len(test_loader)
    test_accuracy = 100 * correct / total

    auc = roc_auc_score(all_labels, all_probs)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
    print(f'Total: {total}, Correct: {correct}')
    print(f'F1 Score: {f1:.4f}, ROC AUC: {auc:.4f}')

    metrics = calculate_fairness_metrics(all_labels, all_preds, all_ids)
    equal_accuracy, equal_opportunity, equalized_odds, disparate_impact, demographic_parity, treatment_equality, test_fairness, conditional_statistical_parity, male_tp, male_fp, male_tn, male_fn, female_tp, female_fp, female_tn, female_fn = metrics

    print(f'Equal Accuracy: {equal_accuracy:.4f}')
    print(f'Equal Opportunity: {equal_opportunity:.4f}')
    print(f'Equalized Odds: {equalized_odds:.4f}')
    print(f'Disparate Impact: {disparate_impact:.4f}')
    print(f'Demographic Parity: {demographic_parity:.4f}')
    print(f'Treatment Equality: {treatment_equality:.4f}')
    print(f'Test Fairness: {test_fairness:.4f}')
    print(f'Conditional Statistical Parity: {conditional_statistical_parity:.4f}')
    print(f'Male - TP: {male_tp}, FP: {male_fp}, TN: {male_tn}, FN: {male_fn}')
    print(f'Female - TP: {female_tp}, FP: {female_fp}, TN: {female_tn}, FN: {female_fn}')

    return test_accuracy, f1, auc, equal_accuracy, equal_opportunity, equalized_odds, disparate_impact, demographic_parity, treatment_equality, test_fairness, conditional_statistical_parity

def print_class_distribution(data_dir):
    data = datasets.ImageFolder(root=data_dir, transform=transform)
    class_counts = {class_name: 0 for class_name in data.classes}
    
    for _, label in data.samples:
        class_name = data.classes[label]
        class_counts[class_name] += 1
    
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count}")

def print_gender_distribution(data_dir):
    """
    Training data gender distribution:
        Male: 20630
        Female: 27384
    Test data gender distribution:
        Male: 4817
        Female: 4883
    """
    data = datasets.ImageFolder(root=data_dir, transform=transform)
    male_count = 0
    female_count = 0
    
    for path, _ in data.samples:
        filename = os.path.basename(path)
        id = filename[:5]
        if id in male_ids:
            male_count += 1
        elif id in female_ids:
            female_count += 1

    print(f"Male: {male_count}")
    print(f"Female: {female_count}")

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

    train_dir = os.path.join(base_dir, '../data/train-aug')
    test_dir = os.path.join(base_dir, '../data/test')

    print("Training data class distribution:")
    print_class_distribution(train_dir)

    print("Test data class distribution:")
    print_class_distribution(test_dir)

    print("Training data gender distribution:")
    print_gender_distribution(train_dir)

    print("Test data gender distribution:")
    print_gender_distribution(test_dir)

    print("Evaluating model...", model_type)
    evaluate_model(model, device, base_dir)