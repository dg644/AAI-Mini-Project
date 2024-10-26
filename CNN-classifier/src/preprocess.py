import os
import glob
import shutil
import random
from tqdm import tqdm

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
    image_list = os.listdir(image_dir)
    for image in tqdm(image_list, desc="Processing images by person"):
        person_id = image.split('t')[0]  # assuming person id is before 't' in filename
        if person_id not in person_images:
            person_images[person_id] = []
        person_images[person_id].append(image)
    
    train_images = []
    val_images = []
    test_images = []
    random.seed = 42
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

    # ensure the directories are empty
    for dir_path in [train_pain_dir, train_no_pain_dir, val_pain_dir, val_no_pain_dir, test_pain_dir, test_no_pain_dir]:
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    train_pain_images, val_pain_images, test_pain_images = split_by_person(pain_dir)
    train_no_pain_images, val_no_pain_images, test_no_pain_images = split_by_person(no_pain_dir)
    
    for image in tqdm(train_pain_images, desc="Copying train pain images"):
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
    
    print("Split complete!\n")



def main():
    dataset_dir = r'C:\Users\dylan\OneDrive - University of Cambridge\Part II\Affective AI\Project' # change this to the path of the dataset
    project_dir = r'C:\Users\dylan\work\AAI-Mini-Project\CNN-classifier' # change this to the path of the project
    image_dir = 'UNBC/Images'
    label_dir = 'UNBC/Frame_Labels/PSPI'
    train_dir = 'data/train'
    val_dir = 'data/val'
    test_dir = 'data/test'

    # Example usage of label_images
    # labels = label_images(label_dir, dataset_dir)
    # print(labels['ll042t1aaaff001.png']) # 0 = no pain (PSPI = 0)
    # print(labels['ll042t1aaaff022.png']) # 1 = pain (PSPI = 2)

    # uncomment to run sort_images
    sort_images(image_dir, label_dir, project_dir, dataset_dir)

    # uncomment to run split_images
    split_images(project_dir, train_dir, val_dir, test_dir)

if __name__ == '__main__':
    main()