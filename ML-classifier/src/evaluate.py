import os
import numpy as np
from tqdm import tqdm
from skimage.feature import hog
from sklearn.metrics import f1_score, roc_auc_score
import pickle
from PIL import Image
from pathlib import Path

def split_images_male_female(image_files, male_ids, female_ids):
    male_array = []
    female_array = []
    for file in image_files:
        person_id = file[0:5]  # assuming person id is first 5 characters in image filename
        if person_id in male_ids:
            male_array.append(file)
        elif person_id in female_ids:
            female_array.append(file)
        else:
            raise ValueError("Person id not found", person_id)
    return male_array, female_array


###NOTE: this is only slightly different to the one in sgdtrain.py, it takes a list of image filenames, instead of an image directory
def feature_extract(image_filenames, image_dir, feature_array_filepath):

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
    image_subfolder = str(Path(image_dir).parts[-2:]).replace("'", "").replace("(", "").replace(")", "").replace(", ", "/")
    for file in tqdm(image_filenames, desc="Feature extraction on " + image_subfolder):
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
    
def main():
    # Get the absolute path of the current script’s directory
    project_dir = "/Users/yutingshang/Documents/Projects/AAI-Mini-Project/ML-classifier"
    test_dir = 'data/test'
    test_pain_image_dir = os.path.join(project_dir, test_dir, 'pain')
    test_no_pain_image_dir = os.path.join(project_dir, test_dir, 'no-pain')

    #define the male and female ids
    male_ids = ["048-aa048", "052-dr052", "064-ak064", "095-tv095","096-bg096", "097-gf097",
                "101-mg101","103-jk103","109-ib109","115-jy115","120-kz120","123-jh123"]

    female_ids = ["042-ll042", "043-jh043", "047-jl047", "049-bm049", "059-fn059", "066-mg066", 
                  "080-bn080", "092-ch092", "106-nm106", "107-hs107", "108-th108", "121-vw121", "124-dn124"]

    male_ids = [id.split('-')[1] for id in male_ids]    #extract the id from the full id, after the -
    female_ids = [id.split('-')[1] for id in female_ids]    


    #Splits the male female data when splitting into arrays
    test_pain_image_files = os.listdir(os.path.join(project_dir, test_dir, 'pain'))
    test_no_pain_image_files = os.listdir(os.path.join(project_dir, test_dir, 'no-pain'))

    male_pain_files, female_pain_files = split_images_male_female(test_pain_image_files, male_ids, female_ids)
    male_no_pain_files, female_no_pain_files = split_images_male_female(test_no_pain_image_files, male_ids, female_ids)

    #filepaths for the feature arrays
    test_pain_array_filepath_M = os.path.join(project_dir, 'processed-array', 'test-pain-hog-male')
    test_pain_array_filepath_F = os.path.join(project_dir, 'processed-array', 'test-pain-hog-female')
    test_no_pain_array_filepath_M = os.path.join(project_dir, 'processed-array', 'test-no-pain-hog-male')
    test_no_pain_array_filepath_F = os.path.join(project_dir, 'processed-array', 'test-no-pain-hog-female')

    #Extract features
    test_pain_numpy_arrays_M = feature_extract(male_pain_files, test_pain_image_dir, test_pain_array_filepath_M)
    test_pain_numpy_arrays_F = feature_extract(female_pain_files, test_pain_image_dir, test_pain_array_filepath_F)
    test_no_pain_numpy_arrays_M = feature_extract(male_no_pain_files, test_no_pain_image_dir, test_no_pain_array_filepath_M)
    test_no_pain_numpy_arrays_F = feature_extract(female_no_pain_files, test_no_pain_image_dir, test_no_pain_array_filepath_F)

    # Combine the pain and no-pain images into a single array, and create a corresponding array of labels 1=pain, 0=no pain
    test_image_numpy_data_male = np.concatenate([test_pain_numpy_arrays_M, test_no_pain_numpy_arrays_M])
    test_label_arrays_male = np.concatenate([np.ones(len(test_pain_numpy_arrays_M)), np.zeros(len(test_no_pain_numpy_arrays_M))])

    test_image_numpy_data_female = np.concatenate([test_pain_numpy_arrays_F, test_no_pain_numpy_arrays_F])
    test_label_arrays_female = np.concatenate([np.ones(len(test_pain_numpy_arrays_F)), np.zeros(len(test_no_pain_numpy_arrays_F))])

    print("male images shape:", len(test_image_numpy_data_male))
    print("female images shape:", len(test_image_numpy_data_female))


    # Load trained model from file
    with open(os.path.join(project_dir, 'model','rbf-on-augmented-data-1.pkl'), 'rb') as f:
        classifer = pickle.load(f)
                                    
    predicted_pain_array_M = classifer.predict(test_image_numpy_data_male)
    accuracy_M = np.mean(predicted_pain_array_M == test_label_arrays_male)*100
    f1_M = f1_score(test_label_arrays_male, predicted_pain_array_M, average='weighted')    #f1 score for 2 classes combined
    roc_auc_M = roc_auc_score(test_label_arrays_male, predicted_pain_array_M) 

    predicted_pain_array_F = classifer.predict(test_image_numpy_data_female)
    accuracy_F = np.mean(predicted_pain_array_F == test_label_arrays_female)*100
    f1_F = f1_score(test_label_arrays_female, predicted_pain_array_F, average='weighted')    #f1 score for 2 classes combined
    roc_auc_F = roc_auc_score(test_label_arrays_female, predicted_pain_array_F)

    print("Accuracy Male:", accuracy_M)
    print("Accuracy Female:", accuracy_F)
    print("F1 score Male:", f1_M)
    print("F1 score Female:", f1_F)
    print("ROC AUC score Male:", roc_auc_M)
    print("ROC AUC score Female:", roc_auc_F)


if __name__ == "__main__":
    main()