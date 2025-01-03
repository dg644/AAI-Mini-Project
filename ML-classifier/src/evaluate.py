import os
import numpy as np
from tqdm import tqdm
from skimage.feature import hog
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, hinge_loss
# from aif360.metrics import equal_opportunity_difference, average_odds_difference, disparate_impact_ratio, statistical_parity_difference
import pickle
from PIL import Image
from pathlib import Path
from sklearn.metrics import confusion_matrix


# copied from training scripts
def feature_extraction(image_dir, feature_array_filepath):

    feature_arrays = []    #features for all images

    if (os.path.isfile(feature_array_filepath)):    #array already exists
        subfolder_path = str(Path(feature_array_filepath).parts[-2:]).replace("'", "").replace("(", "").replace(")", "").replace(", ", "/")   #gets the last 2 subdirs, and turns tuple back to string
        proceed = input("Feature array already exists in " + subfolder_path + ", do you want to re-extract features? (y/n)")
        while proceed != "y" and proceed != "n":
            proceed = input("Invalid input, do you want to extract features again? (y/n)")
        if proceed == "n":
            with open(feature_array_filepath, 'rb') as f:
                feature_arrays = pickle.load(f)
            return feature_arrays
        
    #otherwise, extract features

    #check the image directory exists
    if not os.path.exists(image_dir):
        print("ERROR: Image directory does not exist: " + image_dir)
        return

    image_files = os.listdir(image_dir)   #get all the image files in the directory

    image_subfolder = str(Path(image_dir).parts[-2:]).replace("'", "").replace("(", "").replace(")", "").replace(", ", "/")
    for file in tqdm(image_files, desc="Feature extraction on " + image_subfolder):
        image = Image.open(os.path.join(image_dir, file))
        image = image.convert('L')   #convert to grayscale for HOG
        image = image.resize((224,224))
        image = np.array(image)
        hog_features = hog(image, pixels_per_cell=(16,16),     #Then number of features = 13*13*2*2*9 = 6,084 features (where 13 = 224/16-1), which is a good reduction 
        cells_per_block=(2, 2), 
        orientations=9, 
        block_norm='L2-Hys')  # extract HOG features - also flattens it to 1D array
        feature_arrays.append(hog_features)  

    os.makedirs(os.path.dirname(feature_array_filepath), exist_ok=True)
    with open(feature_array_filepath, 'wb') as f:
        pickle.dump(feature_arrays, f)

    return feature_arrays

# get the ids for all the images in a directory
def get_ids(image_dir):

     #check the image directory exists
    if not os.path.exists(image_dir):
        print("ERROR: Image directory does not exist: " + image_dir)
        return
    
    image_files = os.listdir(image_dir)   #get all the image files in the directory
    ids = [file[0:5] for file in image_files]   #extract the id from the full id, after the -
    return ids


def print_gender_distribution(image_dir, male_ids, female_ids):
    """
    Training data gender distribution:
        Male: 20630
        Female: 27384
    Test data gender distribution:
        Male: 4817
        Female: 4883
    """

    pain_image_files = os.listdir(os.path.join(image_dir, 'pain'))   #get all the image files in the directory
    no_pain_image_files = os.listdir(os.path.join(image_dir, 'no-pain'))   #get all the image files in the directory
    image_files = np.concatenate([pain_image_files, no_pain_image_files])
    ids = [file[0:5] for file in image_files]   #extract the id from the full id, after the -
    male_count = sum([1 for id in ids if id in male_ids])
    female_count = sum([1 for id in ids if id in female_ids])
    print(f"Male: {male_count}")
    print(f"Female: {female_count}")

    
def calculate_fairness_metrics(labels, preds, ids, male_ids, female_ids):
    # print(f"Length of labels: {len(labels)}")
    # print(f"Length of preds: {len(preds)}")
    # print(f"Length of ids: {len(ids)}")

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
    # conditional_statistical_parity = abs(male_positive_rate - female_positive_rate)



    # Confusion matrix details
    male_tp = male_cm[1, 1]
    male_fp = male_cm[0, 1]
    male_tn = male_cm[0, 0]
    male_fn = male_cm[1, 0]

    female_tp = female_cm[1, 1]
    female_fp = female_cm[0, 1]
    female_tn = female_cm[0, 0]
    female_fn = female_cm[1, 0]

    return equal_accuracy, equal_opportunity, equalized_odds, disparate_impact, demographic_parity, treatment_equality, test_fairness, male_tp, male_fp, male_tn, male_fn, female_tp, female_fp, female_tn, female_fn


def main():
    # Get the absolute path of the current script’s directory
    project_dir = "/Users/yutingshang/Documents/Projects/AAI-Mini-Project/ML-classifier"
    train_dir = os.path.join(project_dir, 'data', 'train-aug')
    test_dir = os.path.join(project_dir, 'data', 'test')
    test_pain_image_dir = os.path.join(test_dir, 'pain')
    test_no_pain_image_dir = os.path.join(test_dir, 'no-pain')

    #define the male and female ids
    male_ids = ["048-aa048", "052-dr052", "064-ak064", "095-tv095","096-bg096", "097-gf097",
                "101-mg101","103-jk103","109-ib109","115-jy115","120-kz120","123-jh123"]

    female_ids = ["042-ll042", "043-jh043", "047-jl047", "049-bm049", "059-fn059", "066-mg066", 
                  "080-bn080", "092-ch092", "106-nm106", "107-hs107", "108-th108", "121-vw121", "124-dn124"]

    male_ids = [id.split('-')[1] for id in male_ids]    #extract the id from the full id, after the -
    female_ids = [id.split('-')[1] for id in female_ids]    

    #filepaths for the feature arrays and extract features into arrays
    test_pain_array_filepath = os.path.join(project_dir, 'processed-array', 'test-pain-hog')
    test_no_pain_array_filepath = os.path.join(project_dir, 'processed-array', 'test-no-pain-hog')

    test_pain_array = feature_extraction(test_pain_image_dir, test_pain_array_filepath)
    test_no_pain_array = feature_extraction(test_no_pain_image_dir, test_no_pain_array_filepath)

    #combine the pain and no-pain data and create labels
    test_image_numpy_data = np.concatenate([test_pain_array, test_no_pain_array])
    all_labels = np.concatenate([np.ones(len(test_pain_array)), np.zeros(len(test_no_pain_array))])

    del test_pain_array
    del test_no_pain_array

    # Load trained model from file
    model_filepath = os.path.join(project_dir, 'model','sgd_classifier-on-augmented-data.pkl')
    # model_filepath = os.path.join(project_dir, 'model','rbf-on-augmented-data.pkl')
    if not os.path.isfile(model_filepath):
        print("ERROR: Model file does not exist: " + model_filepath)
        return
    
    with open(model_filepath, 'rb') as f:
        classifier = pickle.load(f)

    #predict the pain
    print("\nPredicting...")
    all_predictions = classifier.predict(test_image_numpy_data)

    #overall performance metrics
    ###NOTE: Hinge loss is the loss function used for the SGDClassifier
    accuracy = accuracy_score(all_labels, all_predictions)
    loss = hinge_loss(all_labels, all_predictions)    
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    roc_auc = roc_auc_score(all_labels, all_predictions)

    print("\nTest accuracy:", accuracy)
    print("Hinge loss:", loss)
    print(f"Total: {len(all_labels)} Correct: {sum(all_labels == all_predictions)}")
    print("F1 score:", f1)
    print("ROC AUC score:", roc_auc)

    #get the ids for all the images in the order of the test data
    all_ids = np.concatenate([get_ids(test_pain_image_dir), get_ids(test_no_pain_image_dir)])   

    #fairness metrics      
    metrics = calculate_fairness_metrics(all_labels, all_predictions, all_ids, male_ids, female_ids)
    equal_accuracy, equal_opportunity, equalized_odds, disparate_impact, demographic_parity, treatment_equality, test_fairness, male_tp, male_fp, male_tn, male_fn, female_tp, female_fp, female_tn, female_fn = metrics

    print(f'\nEqual Accuracy: {equal_accuracy:.4f}')
    print(f'Equal Opportunity: {equal_opportunity:.4f}')
    print(f'Equalized Odds: {equalized_odds:.4f}')
    print(f'Disparate Impact: {disparate_impact:.4f}')
    print(f'Demographic Parity: {demographic_parity:.4f}')
    print(f'Treatment Equality: {treatment_equality:.4f}')
    print(f'Test Fairness: {test_fairness:.4f}')
    # print(f'Conditional Statistical Parity: {conditional_statistical_parity:.4f}')

    print("\nConfusion matrix:")
    print(f'Male - TP: {male_tp}, FP: {male_fp}, TN: {male_tn}, FN: {male_fn}')
    print(f'Female - TP: {female_tp}, FP: {female_fp}, TN: {female_tn}, FN: {female_fn}')   

    print("\nTraining data gender distribution:")
    print_gender_distribution(train_dir, male_ids, female_ids)

    print("Test data gender distribution:")
    print_gender_distribution(test_dir, male_ids, female_ids)

    validation_dir = os.path.join(project_dir, 'data', 'val')
    print("Training pain data count:")
    print(len(os.listdir(os.path.join(train_dir, 'pain'))))
    print("Training no-pain data count:")
    print(len(os.listdir(os.path.join(train_dir, 'no-pain'))))
    print("Validation pain data count:")
    print(len(os.listdir(os.path.join(validation_dir, 'pain'))))
    print("Validation no-pain data count:")
    print(len(os.listdir(os.path.join(validation_dir, 'no-pain'))))
    print("Test pain data count:")
    print(len(os.listdir(test_pain_image_dir)))
    print("Test no-pain data count:")
    print(len(os.listdir(test_no_pain_image_dir)))

    print("Feature vector dimension")
    print(len(test_image_numpy_data[0]))

    print("Original training data gender distribution:")
    orig_train_dir = os.path.join(project_dir, 'data', 'train')
    print_gender_distribution(orig_train_dir, male_ids, female_ids)


if __name__ == "__main__":
    main()