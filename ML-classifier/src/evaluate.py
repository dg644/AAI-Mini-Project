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

test_pain_numpy_arrays = []
test_no_pain_numpy_arrays = []

if not(os.path.exists(os.path.join(project_dir, 'processed-array', 'pain-test-array')) and os.path.exists(os.path.join(project_dir, 'processed-array', 'no-pain-test-array'))):
    test_pain_image_files = os.listdir(os.path.join(project_dir, 'data', 'test', 'pain'))
    test_no_pain_image_files = os.listdir(os.path.join(project_dir, 'data', 'test', 'no-pain'))

    for file in tqdm(test_pain_image_files, desc="Converting test pain images"):
        image = imread(os.path.join(project_dir, 'data', 'test', 'pain', file), as_gray=True)
        image = rescale(image, (224/image.shape[0], 224/image.shape[1]), mode='edge') # resize the image to 224x224 by extending the edge pixels
        image_hog = hog(image, pixels_per_cell=(14,14), 
        cells_per_block=(2, 2), 
        orientations=9, 
        block_norm='L2-Hys')  # extract HOG features - already flattened. 
        test_pain_numpy_arrays.append(image_hog)

    for file in tqdm(test_no_pain_image_files, desc="Converting test no pain images"):
        image = imread(os.path.join(project_dir, 'data', 'test', 'no-pain', file), as_gray=True)
        image = rescale(image, (224/image.shape[0], 224/image.shape[1]), mode='edge') # resize the image to 224x224 by extending the edgre pixels
        image_hog = hog(image, pixels_per_cell=(14,14), 
        cells_per_block=(2, 2), 
        orientations=9, 
        block_norm='L2-Hys')  # extract HOG features - already flattened. Number of features = 15*15*2*2*9 = 8100   (15 comes from 224/14 = 16; 16-1 = 15)
        test_no_pain_numpy_arrays.append(image_hog)

    # Save the test arrays to a file
    os.makedirs(os.path.join(project_dir, 'processed-array'), exist_ok=True)
    with open(os.path.join(project_dir, 'processed-array','pain-test-array'), 'wb') as f:
        pickle.dump(test_pain_numpy_arrays, f)
    with open(os.path.join(project_dir, 'processed-array','no-pain-test-array'), 'wb') as f:
        pickle.dump(test_no_pain_numpy_arrays, f)
else:
    # Load the arrays from a file
    with open(os.path.join(project_dir, 'processed-array','pain-test-array'), 'rb') as f:
        test_pain_numpy_arrays = pickle.load(f)
    with open(os.path.join(project_dir, 'processed-array','no-pain-test-array'), 'rb') as f:
        test_no_pain_numpy_arrays = pickle.load(f)

# Combine the pain and no-pain images into a single array, and create a corresponding array of labels 1=pain, 0=no pain
test_image_numpy_data = np.concatenate([test_pain_numpy_arrays, test_no_pain_numpy_arrays])
test_label_arrays = np.concatenate([np.ones(len(test_pain_numpy_arrays)), np.zeros(len(test_no_pain_numpy_arrays))])


# Load trained model from file
with open(os.path.join(project_dir, 'model','sgd_classifier-augment.pkl'), 'rb') as f:
    sgd_clf = pickle.load(f)
                                
predicted_pain_array = sgd_clf.predict(test_image_numpy_data)
accuracy = np.mean(predicted_pain_array == test_label_arrays)*100
f1 = f1_score(test_label_arrays, predicted_pain_array, average='weighted')    #f1 score for 2 classes combined
roc_auc = roc_auc_score(test_label_arrays, predicted_pain_array) 

print("Accuracy:", accuracy)
print("F1 score:", f1)
print("ROC AUC score:", roc_auc)