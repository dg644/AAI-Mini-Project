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
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier
import pickle
from PIL import Image
from pathlib import Path


# takes the input_dir images, feature extracts for all the images in the directory
# returns an array of the features for each image
# saves a pickled version in the array_dir
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


def hyperparameter_tuning_rbf(train_feature_array, val_feature_array, train_labels, val_labels, project_dir):
    print("\nHyperparameter tuning on training and validation data")

    #combine the training and validation data
    train_val_feature_array = np.concatenate([train_feature_array, val_feature_array])
    train_val_labels = np.concatenate([train_labels, val_labels])    #concatenates pain and no-pain labels

    #tag the training data as -1, and validation data as 0 for the GridSearchCV
    split_indicator = np.concatenate([np.ones(len(train_feature_array)) * -1, np.zeros(len(val_feature_array))])    

    ps = PredefinedSplit(test_fold=split_indicator)

    #define the parameter grid
    param_grid = {
        'sgd__alpha': [0.00001, 0.000001, 0.0000001],   #Regularisation, for generalisation
        'nystroem__gamma': [0.001, 0.0001, 0.00001],   #Gamma for the RBF kernel
        'nystroem__n_components': [1000, 2000, 5000],   #Number of features nystroem approximates
    }

    #pipeline with Nystroem and a linear SVM (SGDClassifier)
    pipeline = Pipeline([('scaler', StandardScaler()), 
                         ('nystroem', Nystroem(kernel='rbf', random_state=42)),
                         ('sgd', SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3, random_state=42, verbose=0))
                         ])

    #perform hyperparameter tuning on the Nystroem and SGDClassifier searching the parameter grid
    #cross validation is done using the PredefinedSplit (preset the train and validation data)
    #maximise the accuracy to find the best hyperparameters, higher val accuracy ~ lower val cross entropy loss
    #refit: do not refit the model, since this is only one-fold cross validation and uses PredefinedSplit
    grid_search = GridSearchCV(pipeline, param_grid, cv=ps, scoring='accuracy', n_jobs=1, refit=False, verbose=3)
    grid_search.fit(train_val_feature_array, train_val_labels)

    #append it to the hyperparameters file just as a copy
    os.makedirs(os.path.dirname(os.path.join(project_dir, 'src', 'hyperparameters')), exist_ok=True)
    with open(os.path.join(project_dir, 'src', 'hyperparameters', 'rbf-best-params.txt'), 'a') as f:
        f.write("\n" + str(grid_search.best_params_))

    return grid_search.best_params_

def train_rbf(train_feature_array, train_labels, best_params, model_save_filepath):

    if os.path.isfile(model_save_filepath):    #model with same filepath already exists
        proceed = input("RBF model already exists, do you want to delete it and proceed? (y/n)")
        while proceed != "y" and proceed != "n":
            proceed = input("Invalid input, do you want to train the model again? (y/n)")
        if proceed == "n":
            return     #return, don't save the model
    
    print("\nTraining RBF SVM (Nystroem + SGDClassifier) and saving the model")

    #create the pipeline again with the feature scaler
    #add the best parameters
    best_nystroem_params = {key.split('__')[1]: value for key, value in best_params.items() if key.startswith('nystroem')}   #remove the nystroem__ prefix
    best_sgd_params = {key.split('__')[1]: value for key, value in best_params.items() if key.startswith('sgd')}   #remove the sgd__ prefix
    pipeline = Pipeline([('scaler', StandardScaler()), 
                         ('nystroem', Nystroem(**best_nystroem_params, kernel='rbf', random_state=42)),
                         ('sgd', SGDClassifier(**best_sgd_params, loss='hinge', max_iter=1000, tol=1e-3, random_state=42, verbose=0))
                         ])
    
    #train the model
    pipeline.fit(train_feature_array, train_labels)

    #save the entire pipeline - so the same scaling parameters can be applied to the test data   
    os.makedirs(os.path.dirname(model_save_filepath), exist_ok=True)
    with open(model_save_filepath, 'wb') as f:
        pickle.dump(pipeline, f)


def main():
    # Get the absolute path of the project directory
    project_dir = "/Users/yutingshang/Documents/Projects/AAI-Mini-Project/ML-classifier"

    ###NOTE: using the augmented training data
    train_dir = 'data/train-aug'
    val_dir = 'data/val'
    train_pain_image_dir = os.path.join(project_dir, train_dir, 'pain')
    train_no_pain_image_dir = os.path.join(project_dir, train_dir, 'no-pain')
    val_pain_image_dir = os.path.join(project_dir, val_dir, 'pain')
    val_no_pain_image_dir = os.path.join(project_dir, val_dir, 'no-pain')

    #UNCOMMENT TO VISUALISE HOG FEATURE
    # sample_image = os.listdir(train_no_pain_image_dir)[6]    #pick a sample image
    # visualise_hog_feature(os.path.join(train_no_pain_image_dir, sample_image))

    # filepaths for the feature arrays
    train_pain_array_filepath = os.path.join(project_dir, 'processed-array', 'train-pain-hog')
    train_no_pain_array_filepath = os.path.join(project_dir, 'processed-array', 'train-no-pain-hog')
    val_pain_array_filepath = os.path.join(project_dir, 'processed-array', 'val-pain-hog')
    val_no_pain_array_filepath = os.path.join(project_dir, 'processed-array', 'val-no-pain-hog')

    # feature extraction, save the feature arrays
    train_pain_array = feature_extraction(train_pain_image_dir, train_pain_array_filepath)
    train_no_pain_array = feature_extraction(train_no_pain_image_dir, train_no_pain_array_filepath)
    val_pain_array = feature_extraction(val_pain_image_dir, val_pain_array_filepath)
    val_no_pain_array = feature_extraction(val_no_pain_image_dir, val_no_pain_array_filepath)

    del train_pain_image_dir
    del train_no_pain_image_dir
    del val_pain_image_dir
    del val_no_pain_image_dir

    # label the training data
    # 1 = pain, 0 = no pain
    train_pain_labels = np.ones(len(train_pain_array))
    train_no_pain_labels = np.zeros(len(train_no_pain_array))
    val_pain_labels = np.ones(len(val_pain_array))
    val_no_pain_labels = np.zeros(len(val_no_pain_array))

    #combine the pain and no-pain data
    training_array = np.concatenate([train_pain_array, train_no_pain_array])
    training_labels = np.concatenate([train_pain_labels, train_no_pain_labels])
    validation_array = np.concatenate([val_pain_array, val_no_pain_array])
    validation_labels = np.concatenate([val_pain_labels, val_no_pain_labels])

    del train_pain_array
    del train_no_pain_array
    del val_pain_array
    del val_no_pain_array

    #perform hyperparameter tuning on training+validation data
    best_params = hyperparameter_tuning_rbf(training_array, validation_array, training_labels, validation_labels, project_dir)
    print("Best parameters:", best_params)     #strip the prefixes from the best parameters later in the train_rbf function

    #train the model only on the training data, and save it
    model_save_filepath = os.path.join(project_dir, 'model', 'rbf-on-augmented-data.pkl')
    train_rbf(training_array, training_labels, best_params, model_save_filepath)


if __name__ == "__main__":
    main()







