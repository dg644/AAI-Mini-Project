# To train model :
# If you have changed the data, delete the preprocesssed arrays and the model file (or rename) and retrain
# If you have changed the model, you can delete/add the model and retrain - can use preprocesssed arrays
# If you have not changed anything, you can skip the training step - go to evaluate.py to use pickled model
# NOTE : The preprocessed arrays have training or testing data arrays
################

import os
import numpy as np
from tqdm import tqdm
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import rescale
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
import pickle


# Get the absolute path of the current script’s directory
project_dir = "/Users/yutingshang/Documents/AAI-Mini-Project/ML-classifier"

pain_numpy_arrays = []
no_pain_numpy_arrays = []

if not(os.path.exists(os.path.join(project_dir, 'processed-array', 'pain-train-array')) and os.path.exists(os.path.join(project_dir, 'processed-array', 'no-pain-train-array'))):
    # Get the list of image files in the train/pain and train/no-pain directories
    pain_image_files = os.listdir(os.path.join(project_dir, 'data', 'train', 'pain'))
    no_pain_image_files = os.listdir(os.path.join(project_dir, 'data', 'train', 'no-pain'))

    # Load the images as Numpy arrays MxNx3 (3 channels for RGB)
    for file in tqdm(pain_image_files, desc="Converting pain images"):
        image = imread(os.path.join(project_dir, 'data', 'train', 'pain', file), as_gray=True)
        image = rescale(image, (224/image.shape[0], 224/image.shape[1]), mode='edge') # resize the image to 224x224 by extending the edge pixels
        image_hog = hog(image, pixels_per_cell=(14,14), 
        cells_per_block=(2, 2), 
        orientations=9, 
        block_norm='L2-Hys')  # extract HOG features - already flattened. 
        pain_numpy_arrays.append(image_hog)  
        
    for file in tqdm(no_pain_image_files, desc="Converting no pain images"):
        image = imread(os.path.join(project_dir, 'data', 'train', 'no-pain', file), as_gray=True)
        ###QUESTION: why am I not using resize(224,224) here?
        image = rescale(image, (224/image.shape[0], 224/image.shape[1]), mode='edge') # resize the image to 224x224 by extending the edge pixels
        image_hog = hog(image, pixels_per_cell=(14,14), 
        cells_per_block=(2, 2), 
        orientations=9, 
        block_norm='L2-Hys')  # extract HOG features - already flattened. Number of features = 15*15*2*2*9 = 8100   (15 comes from 224/14 = 16; 16-1 = 15)
        no_pain_numpy_arrays.append(image_hog) 

    # Save the arrays to a file
    os.makedirs(os.path.join(project_dir, 'processed-array'), exist_ok=True)
    with open(os.path.join(project_dir, 'processed-array','pain-train-array'), 'wb') as f:
        pickle.dump(pain_numpy_arrays, f)
    with open(os.path.join(project_dir, 'processed-array','no-pain-train-array'), 'wb') as f:
        pickle.dump(no_pain_numpy_arrays, f)

    del pain_image_files
    del no_pain_image_files
else:
    # Load the arrays from a file
    with open(os.path.join(project_dir, 'processed-array','pain-train-array'), 'rb') as f:
        pain_numpy_arrays = pickle.load(f)
    with open(os.path.join(project_dir, 'processed-array','no-pain-train-array'), 'rb') as f:
        no_pain_numpy_arrays = pickle.load(f)

print("pain_numpy_arrays shape:", np.array(pain_numpy_arrays).shape)
print("no_pain_numpy_arrays shape:", np.array(no_pain_numpy_arrays).shape)
#########################################
 

if not (os.path.exists(os.path.join(project_dir, 'model', 'sgd_classifier.pkl'))):
    sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3, verbose=10)     #verbose to show progress

    # Combine the pain and no-pain images into a single array, and create a corresponding array of labels 1=pain, 0=no pain
    image_numpy_data = np.concatenate([pain_numpy_arrays, no_pain_numpy_arrays])
    label_arrays = np.concatenate([np.ones(len(pain_numpy_arrays)), np.zeros(len(no_pain_numpy_arrays))])

    # Check the shapes of the arrays
    print("image_numpy_data shape:", image_numpy_data.shape)
    print("label_arrays shape:", label_arrays.shape)

    del pain_numpy_arrays
    del no_pain_numpy_arrays

    # for _ in tqdm(range(1), desc="Training"):
    sgd_clf.fit(image_numpy_data, label_arrays)

    # Save the trained model to a file
    os.makedirs(os.path.join(project_dir, 'model'), exist_ok=True)
    with open(os.path.join(project_dir, 'model','sgd_classifier-augment.pkl'), 'wb') as f:
        pickle.dump(sgd_clf, f)





