import os
import numpy as np
from tqdm import tqdm
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import rescale
from sklearn.metrics import f1_score, roc_auc_score
import random
import pickle

# Get the absolute path of the current script’s directory
project_dir = "/Users/yutingshang/Documents/AAI-Mini-Project/ML-classifier"

test_pain_numpy_arrays_M = []
test_pain_numpy_arrays_F = []
test_no_pain_numpy_arrays_M = []
test_no_pain_numpy_arrays_F = []

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

#Splits the male female data when splitting into arrays


if not(os.path.exists(os.path.join(project_dir, 'processed-array', 'pain-test-array-male')) and os.path.exists(os.path.join(project_dir, 'processed-array', 'no-pain-test-array-male')) and os.path.exists(os.path.join(project_dir, 'processed-array', 'pain-test-array-female')) and os.path.exists(os.path.join(project_dir, 'processed-array', 'no-pain-test-array-female'))):
    test_pain_image_files = os.listdir(os.path.join(project_dir, 'data', 'test', 'pain'))
    test_no_pain_image_files = os.listdir(os.path.join(project_dir, 'data', 'test', 'no-pain'))

    for file in tqdm(test_pain_image_files, desc="Converting test pain images"):
        image = imread(os.path.join(project_dir, 'data', 'test', 'pain', file), as_gray=True)
        image = rescale(image, (224/image.shape[0], 224/image.shape[1]), mode='edge') # resize the image to 224x224 by extending the edge pixels
        image_hog = hog(image, pixels_per_cell=(14,14), 
        cells_per_block=(2, 2), 
        orientations=9, 
        block_norm='L2-Hys')  # extract HOG features - already flattened. 
        
        person_id = file[0:5]  # assuming person id is first 5 characters in image filename
        if person_id in male_ids:
            test_pain_numpy_arrays_M.append(image_hog)
        elif person_id in female_ids:
            test_pain_numpy_arrays_F.append(image_hog)
        else:
            print("Person id not found", person_id)

    for file in tqdm(test_no_pain_image_files, desc="Converting test no pain images"):
        image = imread(os.path.join(project_dir, 'data', 'test', 'no-pain', file), as_gray=True)
        image = rescale(image, (224/image.shape[0], 224/image.shape[1]), mode='edge') # resize the image to 224x224 by extending the edgre pixels
        image_hog = hog(image, pixels_per_cell=(14,14), 
        cells_per_block=(2, 2), 
        orientations=9, 
        block_norm='L2-Hys')  # extract HOG features - already flattened. Number of features = 15*15*2*2*9 = 8100   (15 comes from 224/14 = 16; 16-1 = 15)

        person_id = file[0:5]  # assuming person id is before 't' in filename
        if person_id in male_ids:
            test_no_pain_numpy_arrays_M.append(image_hog)
        elif person_id in female_ids:
            test_no_pain_numpy_arrays_F.append(image_hog)
        else:
            print("Person id not found", person_id)

    # Save the test arrays to a file
    os.makedirs(os.path.join(project_dir, 'processed-array'), exist_ok=True)
    with open(os.path.join(project_dir, 'processed-array','pain-test-array-male'), 'wb') as f:
        pickle.dump(test_pain_numpy_arrays_M, f)
    with open(os.path.join(project_dir, 'processed-array','pain-test-array-female'), 'wb') as f:
        pickle.dump(test_pain_numpy_arrays_F, f)
    with open(os.path.join(project_dir, 'processed-array','no-pain-test-array-male'), 'wb') as f:
        pickle.dump(test_no_pain_numpy_arrays_M, f)
    with open(os.path.join(project_dir, 'processed-array','no-pain-test-array-female'), 'wb') as f:
        pickle.dump(test_no_pain_numpy_arrays_F, f)
else:
    # Load the arrays from a file
    with open(os.path.join(project_dir, 'processed-array','pain-test-array-male'), 'rb') as f:
        test_pain_numpy_arrays_M = pickle.load(f)
    with open(os.path.join(project_dir, 'processed-array','pain-test-array-female'), 'rb') as f:
        test_pain_numpy_arrays_F = pickle.load(f)
    with open(os.path.join(project_dir, 'processed-array','no-pain-test-array-male'), 'rb') as f:
        test_no_pain_numpy_arrays_M = pickle.load(f)
    with open(os.path.join(project_dir, 'processed-array','no-pain-test-array-female'), 'rb') as f:
        test_no_pain_numpy_arrays_F = pickle.load(f)

# Combine the pain and no-pain images into a single array, and create a corresponding array of labels 1=pain, 0=no pain
test_image_numpy_data_male = np.concatenate([test_pain_numpy_arrays_M, test_no_pain_numpy_arrays_M])
test_label_arrays_male = np.concatenate([np.ones(len(test_pain_numpy_arrays_M)), np.zeros(len(test_no_pain_numpy_arrays_M))])

test_image_numpy_data_female = np.concatenate([test_pain_numpy_arrays_F, test_no_pain_numpy_arrays_F])
test_label_arrays_female = np.concatenate([np.ones(len(test_pain_numpy_arrays_F)), np.zeros(len(test_no_pain_numpy_arrays_F))])

print("male images shape:", len(test_image_numpy_data_male))
print("female images shape:", len(test_image_numpy_data_female))


# Load trained model from file
with open(os.path.join(project_dir, 'model','sgd_classifier-augment.pkl'), 'rb') as f:
    sgd_clf = pickle.load(f)
                                
predicted_pain_array_M = sgd_clf.predict(test_image_numpy_data_male)
accuracy_M = np.mean(predicted_pain_array_M == test_label_arrays_male)*100
f1_M = f1_score(test_label_arrays_male, predicted_pain_array_M, average='weighted')    #f1 score for 2 classes combined
roc_auc_M = roc_auc_score(test_label_arrays_male, predicted_pain_array_M) 

predicted_pain_array_F = sgd_clf.predict(test_image_numpy_data_female)
accuracy_F = np.mean(predicted_pain_array_F == test_label_arrays_female)*100
f1_F = f1_score(test_label_arrays_female, predicted_pain_array_F, average='weighted')    #f1 score for 2 classes combined
roc_auc_F = roc_auc_score(test_label_arrays_female, predicted_pain_array_F)

print("Accuracy Male:", accuracy_M)
print("Accuracy Female:", accuracy_F)
print("F1 score Male:", f1_M)
print("F1 score Female:", f1_F)
print("ROC AUC score Male:", roc_auc_M)
print("ROC AUC score Female:", roc_auc_F)

print("male id", male_ids)
print("female id", female_ids)