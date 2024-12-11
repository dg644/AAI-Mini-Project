import os
import glob
import shutil
import random
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

# gives a dictionary of image filenames and their labels
def label_images(label_dir, dataset_dir):
    label_paths = glob.glob(os.path.join(dataset_dir, label_dir, '*/*/*.txt'))
    print(label_paths)
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


# sort images into pain and no pain folders and then into train and test folders
def sort_images(image_dir, label_dir, project_dir, dataset_dir):
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
    
    print("Sort complete!\n")

def split_by_person(image_dir):
    person_images = {}
    image_list = os.listdir(image_dir)    #this has not guaranteed order
    image_list.sort()
    for image in tqdm(image_list, desc="Processing images by person"):
        person_id = image[:5]  # assuming person id is the first 5 characters of the filename
        if person_id not in person_images:
            person_images[person_id] = []
        person_images[person_id].append(image)
    
    train_images = []
    val_images = []
    test_images = []
    random.seed(42)
    for person_id, images in person_images.items():
        random.shuffle(images)
        train_split_idx = int(0.6 * len(images))
        val_split_idx = int(0.8 * len(images))
        train_images.extend(images[:train_split_idx])
        val_images.extend(images[train_split_idx:val_split_idx])
        test_images.extend(images[val_split_idx:])
    
    return train_images, val_images, test_images

# split images into train and test folders
def split_images(project_dir, train_dir, val_dir, test_dir):
    pain_dir = os.path.join(project_dir, 'sorted-images', 'pain')
    no_pain_dir = os.path.join(project_dir, 'sorted-images', 'no-pain')
    
    train_pain_dir = os.path.join(project_dir, train_dir, 'pain')
    train_no_pain_dir = os.path.join(project_dir, train_dir, 'no-pain')
    val_pain_dir = os.path.join(project_dir, val_dir, 'pain')
    val_no_pain_dir = os.path.join(project_dir, val_dir, 'no-pain')
    test_pain_dir = os.path.join(project_dir, test_dir, 'pain')
    test_no_pain_dir = os.path.join(project_dir, test_dir, 'no-pain')
    
    
    os.makedirs(train_pain_dir, exist_ok=True)
    os.makedirs(train_no_pain_dir, exist_ok=True)
    os.makedirs(val_pain_dir, exist_ok=True)
    os.makedirs(val_no_pain_dir, exist_ok=True)
    os.makedirs(test_pain_dir, exist_ok=True)
    os.makedirs(test_no_pain_dir, exist_ok=True)


    os.makedirs(train_pain_dir, exist_ok=True)
    os.makedirs(train_no_pain_dir, exist_ok=True)
    os.makedirs(val_pain_dir, exist_ok=True)
    os.makedirs(val_no_pain_dir, exist_ok=True)
    os.makedirs(test_pain_dir, exist_ok=True)
    os.makedirs(test_no_pain_dir, exist_ok=True)

    # ensure the directories are empty
    for dir_path in [train_pain_dir, train_no_pain_dir, val_pain_dir, val_no_pain_dir, test_pain_dir, test_no_pain_dir]:
        #recursively delete all files in the directory
        if (os.path.exists(dir_path)):
            shutil.rmtree(dir_path)
        #recreate the directory
        os.makedirs(dir_path, exist_ok=True)

    train_pain_images, val_pain_images, test_pain_images = split_by_person(pain_dir)
    train_no_pain_images, val_no_pain_images, test_no_pain_images = split_by_person(no_pain_dir)

    # duplicate training data for data augmentation, will be flipped horizontally - also using a different 
    # metric for evaluation as the dataset is heavily skewed towards no pain
    ###TODO: edit something here to oversample the data
    for image in tqdm(train_pain_images, desc="Copying train pain images"):
        #not flipped
        
        for i in range(5):        #duplicate 5 times
            name = f"{image.split('.')[0]}_{i}.png"
            shutil.copy(os.path.join(pain_dir, image), os.path.join(train_pain_dir, name))

        # #flipped the duplicate
        # flipped_image = Image.open(os.path.join(pain_dir, image)).transpose(Image.FLIP_LEFT_RIGHT)
        # name = f"{image.split('.')[0]}_{1}.png"
        # flipped_image.save(os.path.join(train_pain_dir, name))
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


    print("Split complete!\n")


#After duplicating the data x5, there is too much pain data now (25060) so we need to randomly remove some to get 24008
def random_undersample(train_pain_dir, train_no_pain_dir):
    pain_images = os.listdir(train_pain_dir)
    pain_images.sort()
    random.seed(42)
    random.shuffle(pain_images)
    diff = len(os.listdir(train_pain_dir)) - len(os.listdir(train_no_pain_dir))
    undersampled_pain_images = pain_images[:diff]
    for image in undersampled_pain_images:
        os.remove(os.path.join(train_pain_dir, image))

def random_augment(train_pain_dir, train_no_pain_dir):
    
    pain_images = os.listdir(train_pain_dir)
    num_pain_images = len(pain_images)
    pain_images.sort()
    non_pain_images = os.listdir(train_no_pain_dir)
    num_non_pain_images = len(non_pain_images)
    non_pain_images.sort()

    
    random.seed(42)

    ###NOTE:randomly flips 1/5 of the images horizontally
    random.shuffle(pain_images)
    random.shuffle(non_pain_images)
    for pain_image in tqdm(pain_images[:num_pain_images//5], desc="Flipping pain images"):
        # print("Flipping", os.path.join(train_pain_dir, pain_image))
        flipped_image = Image.open(os.path.join(train_pain_dir, pain_image)).transpose(Image.FLIP_LEFT_RIGHT)
        flipped_image.save(os.path.join(train_pain_dir, pain_image))   #save back to the same file

    for non_pain_image in tqdm(non_pain_images[:num_non_pain_images//5], desc="Flipping non pain images"):
        flipped_image = Image.open(os.path.join(train_no_pain_dir, non_pain_image)).transpose(Image.FLIP_LEFT_RIGHT)
        flipped_image.save(os.path.join(train_no_pain_dir, non_pain_image))

    ###NOTE:randomly rotates 1/5 of the images by (-20, 20) degrees
    random.shuffle(pain_images)
    random.shuffle(non_pain_images)
    for pain_image in tqdm(pain_images[:num_pain_images//5], desc="Rotating pain images"):
        # print("Rotating", os.path.join(train_pain_dir, pain_image))
        rotated_image = Image.open(os.path.join(train_pain_dir, pain_image)).rotate(random.randint(-20,20))
        rotated_image.save(os.path.join(train_pain_dir, pain_image))

    for non_pain_image in tqdm(non_pain_images[:num_non_pain_images//5], desc="Rotating non pain images"):
        rotated_image = Image.open(os.path.join(train_no_pain_dir, non_pain_image)).rotate(random.randint(-20,20))
        rotated_image.save(os.path.join(train_no_pain_dir, non_pain_image))
    
    ###NOTE:randomly crops 1/5 of the images on the longer side to a square
    random.shuffle(pain_images)
    random.shuffle(non_pain_images)
    for pain_image in tqdm(pain_images[:num_pain_images//5], desc="Cropping pain images"):
        # print("Cropping", os.path.join(train_pain_dir, pain_image))
        pain_image_obj = Image.open(os.path.join(train_pain_dir, pain_image))
        length = min(pain_image_obj.size)
        left = random.randint(0, (pain_image_obj.size[0] - length)/2)
        top = 0
        right = left + length
        bottom = length
        pain_image_obj = pain_image_obj.crop((left, top, right, bottom))
        pain_image_obj.save(os.path.join(train_pain_dir, pain_image))

    for non_pain_image in tqdm(non_pain_images[:num_non_pain_images//5], desc="Cropping non pain images"):
        non_pain_image_obj = Image.open(os.path.join(train_no_pain_dir, non_pain_image))
        length = min(non_pain_image_obj.size)
        left = random.randint(0, (non_pain_image_obj.size[0] - length)/2)
        top = 0
        right = left + length
        bottom = length
        non_pain_image_obj = non_pain_image_obj.crop((left, top, right, bottom))
        non_pain_image_obj.save(os.path.join(train_no_pain_dir, non_pain_image))

    
    random.shuffle(pain_images)
    random.shuffle(non_pain_images)
    ###NOTE:changes histogram equalisation for 1/5 of the images, only on the value channel
    for pain_image in tqdm(pain_images[:num_pain_images//5], desc="Histogram equalisation pain images"):
        # print("Histogram equalisation", os.path.join(train_pain_dir, pain_image))
        image = Image.open(os.path.join(train_pain_dir, pain_image))
        image = np.array(image)
        image = rgb2hsv(image)
        image[:,:,2] = exposure.equalize_hist(image[:,:,2])     #equalise the value channel
        image = hsv2rgb(image)
        image = Image.fromarray((image*255).astype(np.uint8))    #scales back to 0-255 and convert to image
        image.save(os.path.join(train_pain_dir, pain_image))
    
    for non_pain_image in tqdm(non_pain_images[:num_non_pain_images//5], desc="Histogram equalisation non pain images"):
        image = Image.open(os.path.join(train_no_pain_dir, non_pain_image))
        image = np.array(image)
        image = rgb2hsv(image)
        image[:,:,2] = exposure.equalize_hist(image[:,:,2])     #equalise the value channel
        image = hsv2rgb(image)
        image = Image.fromarray((image*255).astype(np.uint8))
        image.save(os.path.join(train_no_pain_dir, non_pain_image))





def main():
    dataset_dir = r'/Users/yutingshang/Documents/Projects/AAI-Mini-Project/UNBC_dataset' # change this to the path of the dataset
    project_dir = r'/Users/yutingshang/Documents/Projects/AAI-Mini-Project/ML-classifier' # change this to the path of the project
    image_dir = 'UNBC_dataset/Images'
    label_dir = 'UNBC_dataset/Frame_Labels/PSPI'
    train_dir = 'data/train'
    val_dir = 'data/val'
    test_dir = 'data/test'
    train_pain_dir = os.path.join(project_dir, train_dir, 'pain')
    train_no_pain_dir = os.path.join(project_dir, train_dir, 'no-pain')

    # Example usage of label_images
    # labels = label_images(label_dir, dataset_dir)
    # print(labels['ll042t1aaaff001.png']) # 0 = no pain (PSPI = 0)
    # print(labels['ll042t1aaaff022.png']) # 1 = pain (PSPI = 2)

    # uncomment to run sort_images
    #sort_images(image_dir, label_dir, project_dir, dataset_dir)

    ###NOTE: need to run this again if you have augmented the data!!!
    # uncomment to run split_images
    split_images(project_dir, train_dir, val_dir, test_dir)

    # uncomment to run data augmentation
    random_undersample(train_pain_dir, train_no_pain_dir)   #randomly remove to balance the dataset to both be 24008
    random_augment(train_pain_dir, train_no_pain_dir)   #randomly augment the data


    #uncomment to check number of images
    num_train_pain = len(os.listdir(train_pain_dir))
    num_train_no_pain = len(os.listdir(train_no_pain_dir))
    print(f"Number of training pain images: {num_train_pain}")
    print(f"Number of training no pain images: {num_train_no_pain}")

    # check number of no-pain images is correct
    # total_no_pain = len(os.listdir(os.path.join(project_dir,test_dir,'no-pain')) ) + len(os.listdir(os.path.join(project_dir,train_dir,'no-pain'))) + len(os.listdir(os.path.join(project_dir,val_dir,'no-pain')))
    # print(total_no_pain)

if __name__ == '__main__':
    main()
    