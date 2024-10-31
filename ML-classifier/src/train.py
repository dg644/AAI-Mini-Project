import os
import numpy as np
from tqdm import tqdm
import skimage as ski
from sklearn.ensemble import RandomForestClassifier
import pickle


# Get the absolute path of the current script’s directory
project_dir = "/Users/yutingshang/Documents/AAI-Mini-Project/ML-classifier"

# Get the list of image files in the train/pain and train/no-pain directories
pain_image_files = os.listdir(os.path.join(project_dir, 'data', 'train', 'pain'))
no_pain_image_files = os.listdir(os.path.join(project_dir, 'data', 'train', 'no-pain'))

pain_numpy_arrays = []
no_pain_numpy_arrays = []

# Load the images as Numpy arrays MxNx3 (3 channels for RGB)
for file in tqdm(pain_image_files, desc="Converting pain images"):
    pain_numpy_arrays.append(ski.io.imread(os.path.join(project_dir, 'data', 'train', 'pain', file)).flatten())   #flatten the 3D array
 
for file in tqdm(no_pain_image_files, desc="Converting no pain images"):
    no_pain_numpy_arrays.append(ski.io.imread(os.path.join(project_dir, 'data', 'train', 'no-pain', file)).flatten())

# resize the jageed numpy_arrays to be the same, pad to elements of the 2D array to the maximum length with 0's (black)
max_length = max([len(x) for x in pain_numpy_arrays + no_pain_numpy_arrays])
print(max_length)
padded_pain_array = np.array([np.pad(x, (0, max_length - len(x))) for x in pain_numpy_arrays])
padded_no_pain_array = np.array([np.pad(x, (0, max_length - len(x))) for x in no_pain_numpy_arrays])

del pain_numpy_arrays
del no_pain_numpy_arrays

#########################################
clf = RandomForestClassifier(random_state=0, n_estimators=10, n_jobs=-1, verbose=10)   #verbose to show progress

# Combine the pain and no-pain images into a single array, and create a corresponding array of labels 1=pain, 0=no pain
image_numpy_data = np.concatenate([padded_pain_array, padded_no_pain_array])
label_arrays = np.concatenate([np.ones(len(padded_pain_array)), np.zeros(len(padded_no_pain_array))])

# Check the shapes of the arrays
print("padded_pain_array shape:", padded_pain_array.shape)
print("padded_no_pain_array shape:", padded_no_pain_array.shape)
print("image_numpy_data shape:", image_numpy_data.shape)
print("label_arrays shape:", label_arrays.shape)

del padded_pain_array
del padded_no_pain_array

# for _ in tqdm(range(1), desc="Training"):
clf.fit(image_numpy_data, label_arrays)

# Save the trained model to a file
os.makedirs(os.path.join(project_dir, 'model'), exist_ok=True)
with open(os.path.join(project_dir, 'model','random_forest.pkl'), 'wb') as f:
    pickle.dump(clf, f)



