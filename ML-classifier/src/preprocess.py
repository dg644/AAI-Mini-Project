import os
import glob
import shutil
import random
import math
import warnings
from tqdm import tqdm
from PIL import Image
import numpy as np
from skimage.feature import hog
from skimage import exposure
from imblearn.over_sampling import SMOTE
from skimage.color import rgb2hsv, hsv2rgb

# PSPI 0 = no pain, 1 = trace, 2 = weak, >=3 = strong
# just doing no pain and pain so 0 = no pain, >=1 = pain

# PSPI Score Frequency
# 0 40029
# 1-2 5260
# 3-4 2214
# 5-6 512
# 7-8 132
# 9-10 99
# 11-12 124
# 13-14 23
# 15-16 5

def label_images(label_dir, dataset_dir):

    """
    This function will label the images into pain and no pain
    """

    label_paths = glob.glob(os.path.join(dataset_dir, label_dir, '*/*/*.txt'))
    labels = {}
    for label_path in tqdm(label_paths, desc="Labeling images"):
        with open(label_path, 'r') as file:
            for line in file:
                pspi = line.strip().split(',')[0]
                pspi = int(float(pspi))
                filename = os.path.basename(label_path)
                filename = filename.replace('_facs.txt', '.png')
                if pspi >= 1:
                    labels[filename] = 1 # pain
                else:
                    labels[filename] = 0 # no_pain
    
    print("Labels complete!\n")
    return labels


def sort_images(image_dir, label_dir, project_dir, dataset_dir):

    """
    This function will sort the images into pain and no pain folders
    """

    labels = label_images(label_dir, dataset_dir)
    image_paths = glob.glob(os.path.join(dataset_dir, image_dir, '*/*/*.png'))
    
    for image_path in tqdm(image_paths, desc="Sorting images"):
        filename = os.path.basename(image_path)
        label = labels.get(filename)
        if label:
            dest_dir = os.path.join(project_dir, 'sorted-images', 'pain')
        else:
            dest_dir = os.path.join(project_dir, 'sorted-images', 'no-pain')
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, filename)
        if not os.path.exists(dest_path):
            shutil.copy(image_path, dest_path)
    
    print("Sort complete into pain/no-pain!\n")


def split_by_person(directory):

    """
    This function will split the images into a dictionary of person_id: [list of images]
    """

    person_images = {}     
    image_list = os.listdir(directory)    #this has not guaranteed order
    image_list.sort()
    for image in tqdm(image_list, desc="Processing images by person"):
        person_id = image[:5]  # assuming person id is the first 5 characters of the filename
        if person_id not in person_images:
            person_images[person_id] = []
        person_images[person_id].append(image)
    return person_images


def split_train_val_test_by_person(image_dir):

    """
    This function will split the images into train/val/test, stratified by person
    """

    person_images = split_by_person(image_dir)     #person_images is a dictionary of person_id: [list of images]
    
    train_images = []
    val_images = []
    test_images = []
    random.seed(42)
    for images in person_images.values():
        random.shuffle(images)
        train_split_idx = int(0.6 * len(images))
        val_split_idx = int(0.8 * len(images))
        train_images.extend(images[:train_split_idx])
        val_images.extend(images[train_split_idx:val_split_idx])
        test_images.extend(images[val_split_idx:])
    
    return train_images, val_images, test_images


def split_images(project_dir, train_dir, val_dir, test_dir):

    """
    This function will split the images into train/val/test, pain/no-pain 
    and put them into the correct folders
    """

    pain_dir = os.path.join(project_dir, 'sorted-images', 'pain')
    no_pain_dir = os.path.join(project_dir, 'sorted-images', 'no-pain')
    
    train_pain_dir = os.path.join(project_dir, train_dir, 'pain')
    train_no_pain_dir = os.path.join(project_dir, train_dir, 'no-pain')
    val_pain_dir = os.path.join(project_dir, val_dir, 'pain')
    val_no_pain_dir = os.path.join(project_dir, val_dir, 'no-pain')
    test_pain_dir = os.path.join(project_dir, test_dir, 'pain')
    test_no_pain_dir = os.path.join(project_dir, test_dir, 'no-pain')
    
    # ensure the directories are empty, and create them if they don't exist
    for dir_path in [train_pain_dir, train_no_pain_dir, val_pain_dir, val_no_pain_dir, test_pain_dir, test_no_pain_dir]:
        #recursively delete all files in the directory
        if (os.path.exists(dir_path)):
            shutil.rmtree(dir_path)
        #recreate the directory
        os.makedirs(dir_path, exist_ok=True)

    #split the images into train, val and test, stratified by person
    train_pain_images, val_pain_images, test_pain_images = split_train_val_test_by_person(pain_dir)
    train_no_pain_images, val_no_pain_images, test_no_pain_images = split_train_val_test_by_person(no_pain_dir)

    for image in tqdm(train_pain_images, desc="Copying train pain images"):    #will need to oversample the pain training data
        shutil.copy(os.path.join(pain_dir, image), os.path.join(train_pain_dir, image))
    for image in tqdm(val_pain_images, desc="Copying val pain images"):
        shutil.copy(os.path.join(pain_dir, image), os.path.join(val_pain_dir, image))
    for image in tqdm(test_pain_images, desc="Copying test pain images"):
        shutil.copy(os.path.join(pain_dir, image), os.path.join(test_pain_dir, image))
    for image in tqdm(train_no_pain_images, desc="Copying train no pain images"):
        shutil.copy(os.path.join(no_pain_dir, image), os.path.join(train_no_pain_dir, image))
    for image in tqdm(val_no_pain_images, desc="Copying val no pain images"):
        shutil.copy(os.path.join(no_pain_dir, image), os.path.join(val_no_pain_dir, image))
    for image in tqdm(test_no_pain_images, desc="Copying test no pain images"):
        shutil.copy(os.path.join(no_pain_dir, image), os.path.join(test_no_pain_dir, image))

    print("Split complete into train/val/test, pain/no-pain!\n")



def random_oversample(image_dir, target_num_images):

    """
    This function will duplicate the images to the target number of images 
    Saves the duplicated images back to the same directory, with a suffix of _dup.png

    Usecase: After duplicating the data x4.75, there is still not enough pain data, so we need to randomly duplicate some pain images to get 24008
    """

    images = os.listdir(image_dir)
    images.sort()
    random.seed(42)
    random.shuffle(images)
    diff = target_num_images - len(images)     #number of images to duplicate
    images_to_duplicate = images[:diff]
    print(f"Initial number of pain images: {len(images)}")
    for image in tqdm(images_to_duplicate, desc="Duplicating pain images"):
        new_image_name = f"{image.split('.')[0]}_dup.png"
        shutil.copy(os.path.join(image_dir, image), os.path.join(image_dir, new_image_name))


def random_augment(image_dir):

    """
    This function will randomly augment the images in a directory with random flips, rotations, cropping and histogram equalisation 
    Performed on 1/5 of the images, and saves it back as the same file
    """

    images = os.listdir(image_dir)
    num_images = len(images)
    images.sort()
    
    random.seed(42)

    # randomly flips 1/5 of the images horizontally
    random.shuffle(images)
    for image in tqdm(images[:num_images//5], desc="Flipping images"):
        # print("Flipping", os.path.join(image_dir, image))
        flipped_image = Image.open(os.path.join(image_dir, image)).transpose(Image.FLIP_LEFT_RIGHT)
        flipped_image.save(os.path.join(image_dir, image))   #save back to the same file


    # randomly rotates 1/5 of the images by (-20, 20) degrees
    random.shuffle(images)
    for image in tqdm(images[:num_images//5], desc="Rotating images"):
        # print("Rotating", os.path.join(image_dir, image))
        rotated_image = Image.open(os.path.join(image_dir, image)).rotate(random.randint(-20,20))
        rotated_image.save(os.path.join(image_dir, image))
    
    # randomly crops 1/5 of the images on the longer side to a square
    random.shuffle(images)
    for image in tqdm(images[:num_images//5], desc="Cropping images"):
        # print("Cropping", os.path.join(image_dir, image))
        
        # crops a random 0-5% off left and right side of the image
        image_obj = Image.open(os.path.join(image_dir, image))
        width = image_obj.width
        left = random.randint(0, width//20)
        right = width - random.randint(0, width//20)
        image_obj = image_obj.crop((left, 0, right, image_obj.height))
        image_obj.save(os.path.join(image_dir, image))

    
    random.shuffle(images)
    # changes histogram equalisation for 1/5 of the images, only on the value channel
    for image in tqdm(images[:num_images//5], desc="Histogram equalisation images"):
        # print("Histogram equalisation", os.path.join(image_dir, image))
        image_obj = Image.open(os.path.join(image_dir, image))
        image_obj = image_obj.resize((224, 224))
        image_array = np.array(image_obj)
        image_array = rgb2hsv(image_array)
        image_array[:,:,2] = exposure.equalize_hist(image_array[:,:,2])     #equalise the value channel
        image_array = hsv2rgb(image_array)
        image_obj = Image.fromarray((image_array*255).astype(np.uint8))    #scales back to 0-255 and convert to image
        image_obj.save(os.path.join(image_dir, image))
    

def pain_generate_smote(train_pain_dir, synthetic_train_pain_dir):

    """
    This function will perform SMOTE on the pain images
    Will multiply the number of pain images by 4.75
    and save it to a new directory
    """

    if os.path.exists(synthetic_train_pain_dir):
        shutil.rmtree(synthetic_train_pain_dir)
    os.makedirs(synthetic_train_pain_dir, exist_ok=True)

    #first split the pain images by person, since we will perform SMOTE on each person separately
    people_pain_images = split_by_person(train_pain_dir)
  
    #batch the pain images for each person if it exceeds the batch size, otherwise SMOTE runs out of memory
    batch_size = 200  
    batched_people_pain_images = {}     # new dictionary to store the batched pain images
    for person_id in people_pain_images:
        if len(people_pain_images[person_id]) > batch_size:
            #batch up the images to 300
            for i in range(0, len(people_pain_images[person_id]), batch_size):
                batch = people_pain_images[person_id][i:i+batch_size]
                batched_people_pain_images[person_id+f"_{i//batch_size}"] = batch     #e.g. ak064_0, ak064_1, ak064_2, ...
        else:
            batched_people_pain_images[person_id] = people_pain_images[person_id]

    #for each batch of pain images, perform SMOTE
    total_batches = len(batched_people_pain_images)
    batch_num = 1
    print("Preparing pain images for SMOTE...")
    for person_id, pain_images in batched_people_pain_images.items():
       
        pain_numpy_arrays = []
        num_pain_images = len(pain_images)
        desired_pain_images = math.floor(num_pain_images*4.75)  #desired number of pain images - which are labelled as 1

        #convert the images to arrays of pixels
        print("\n")
        for image in tqdm(pain_images, desc=f"Processing batch #{batch_num}/{total_batches}"):
            image = Image.open(os.path.join(train_pain_dir, image))
            image = image.resize((224, 224))       #resize to 224x224
            image_array = np.array(image)
            pain_numpy_arrays.append(image_array)  
        
        pain_numpy_arrays = np.array(pain_numpy_arrays)   #convert to numpy array
        # print("pain_numpy_arrays shape:", pain_numpy_arrays.shape)     #SANITY CHECK: dimension of pain_numpy_arrays should be {num_pain_images}x224x224x3

        #infer reshape to ({num_pain_images}, 224*224*3) i.e. flattens each image
        pain_numpy_arrays = pain_numpy_arrays.reshape(num_pain_images, -1)     
        # print("pain_numpy_arrays shape after reshape:", pain_numpy_arrays.shape)

        #add a fake majority class sample to the end of the array - just black pixels
        pain_numpy_arrays = np.vstack((pain_numpy_arrays, np.zeros(224*224*3)))  
        labels = np.concatenate((np.ones(len(pain_numpy_arrays)-1), np.zeros(1)))   #add a 0 to the end of the labels for the "non-pain" sample

        #normalise the pixels to 0-1
        pain_numpy_arrays = np.clip(pain_numpy_arrays/255.0, 0, 1)

        #apply smote
        #creates {desired_pain_images} total samples for the pain class (labelled as 1)
        sm = SMOTE(random_state=42, k_neighbors = 10, sampling_strategy={1: desired_pain_images})     
        smote_pain_images = sm.fit_resample(pain_numpy_arrays, labels)  
        pain_labels = smote_pain_images[1]
        smote_pain_images = smote_pain_images[0][pain_labels==1]     #get the pain images labelled as 1, remove the one "non-pain" sample

        #check if the images generated by smote are in the original pain_numpy_arrays images
        mask = (smote_pain_images[:, None] == pain_numpy_arrays).all(-1).any(-1)   
        #get only the synthetic images that are not in the original pain_numpy_arrays
        synthetic_pain_images = smote_pain_images[~mask]      
        print("Actual number of synthetic pain images (non-duplicates):", len(synthetic_pain_images))

        ###NOTE: if I want to only save the synthetic images - use 'synthetic_pain_images' instead of 'smote_pain_images' in later code (i.e. remove this line of code)
        synthetic_pain_images = smote_pain_images    # But in this case, I just want to save all the images generated by smote, including any duplicates

        #convert the images back to 0-255
        synthetic_pain_images = np.clip(synthetic_pain_images*255.0, 0, 255).astype(np.uint8)
        synthetic_pain_images = synthetic_pain_images.reshape(len(synthetic_pain_images), 224, 224, 3)   #reshape to {desired_pain_images}x224x224x3
        
        i=0
        for image in tqdm(synthetic_pain_images, desc=f"Saving {len(synthetic_pain_images)} smote pain images"):
            image = Image.fromarray(image)
            image.save(os.path.join(synthetic_train_pain_dir, f"{person_id}synthetic_{i}.png"))
            i+=1

        batch_num += 1

def resize_non_pain_images(train_no_pain_dir, resized_train_no_pain_dir):

    """
    This function will resize the non-pain images to 224x224
    and save it to a new directory
    """

    if os.path.exists(resized_train_no_pain_dir):
        shutil.rmtree(resized_train_no_pain_dir)
    os.makedirs(resized_train_no_pain_dir, exist_ok=True)

    non_pain_images = os.listdir(train_no_pain_dir)
    for image in tqdm(non_pain_images, desc="Resizing non pain images"):
        image_obj = Image.open(os.path.join(train_no_pain_dir, image))
        image_obj = image_obj.resize((224, 224))
        image_obj.save(os.path.join(resized_train_no_pain_dir, image))






def main():
    dataset_dir = r'/Users/yutingshang/Documents/Projects/AAI-Mini-Project/' # change this to the path of the dataset
    project_dir = r'/Users/yutingshang/Documents/Projects/AAI-Mini-Project/ML-classifier' # change this to the path of the project
    image_dir = 'UNBC_dataset/Images'
    label_dir = 'UNBC_dataset/Frame_Labels/PSPI'
    train_dir = 'data/train'
    train_aug_dir = 'data/train-aug'   #augmented train data
    val_dir = 'data/val'
    test_dir = 'data/test'
    train_pain_dir = os.path.join(project_dir, train_dir, 'pain')
    train_no_pain_dir = os.path.join(project_dir, train_dir, 'no-pain')
    train_aug_pain_dir = os.path.join(project_dir, train_aug_dir, 'pain')
    train_aug_no_pain_dir = os.path.join(project_dir, train_aug_dir, 'no-pain')

    #ignore future warnings about the future of sklearn
    warnings.filterwarnings("ignore", category=FutureWarning)

    # uncomment to run sort_images - it sorts the images into pain and no pain folders
    sort_images(image_dir, label_dir, project_dir, dataset_dir)

    # uncomment to run split_images - it cleans the directories and then splits the pain/no-pain images into train, val and test
    split_images(project_dir, train_dir, val_dir, test_dir)

    if os.path.exists(train_aug_pain_dir):
       
        proceed = input("WARNING: Train augmented pain directory already exists, do you want to delete it and proceed? (y/n)")
        while proceed != "y" and proceed != "n":
            proceed = input("Invalid input, please try again (y/n)")
        if proceed == "n":
            return
        
    # uncomment to run smote, saves the synthetic pain images to a new directory
    pain_generate_smote(train_pain_dir, train_aug_pain_dir)

    # uncomment to run random_oversample, oversamples the pain images to the number of non-pain images - 24007
    random_oversample(train_aug_pain_dir, len(os.listdir(train_no_pain_dir)))

    # uncomment to run resize_non_pain_images, resizes the non-pain images to 224x224 and saves it to a new directory
    resize_non_pain_images(train_no_pain_dir, train_aug_no_pain_dir)

    # uncomment to run data augmentation, rotate, flip, crop and histogram equalisation
    print("Augmenting pain images...")
    random_augment(train_aug_pain_dir)   
    print("Augmenting no pain images...")
    random_augment(train_aug_no_pain_dir)


    #uncomment to check number of images
    num_train_pain = len(os.listdir(train_pain_dir))
    print(f"Number of training pain images: {num_train_pain}")
    num_train_no_pain = len(os.listdir(train_no_pain_dir))
    print(f"Number of training no pain images: {num_train_no_pain}")
    num_synthetic_train_pain = len(os.listdir(train_aug_pain_dir))
    print(f"Number of synthetic training pain images: {num_synthetic_train_pain}")
    num_resized_train_no_pain = len(os.listdir(train_aug_no_pain_dir))
    print(f"Number of resized training no pain images: {num_resized_train_no_pain}")

if __name__ == '__main__':
    main()
    