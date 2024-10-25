import os
import glob
import shutil
import random

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
    for label_path in label_paths:
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
    
    print("Labels complete!")
    return labels



# sort images into pain and no pain folders and then into train and val folders
def sort_images(image_dir, label_dir, project_dir, dataset_dir):
    labels = label_images(label_dir, dataset_dir)
    image_paths = glob.glob(os.path.join(dataset_dir, image_dir, '*/*/*.png'))
    
    for image_path in image_paths:
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
    
    print("Sort complete!")


# splut images into train and val folders
def split_images(project_dir, train_dir, val_dir):
    pain_dir = os.path.join(project_dir, 'sorted-images', 'pain')
    no_pain_dir = os.path.join(project_dir, 'sorted-images', 'no-pain')
    
    train_pain_dir = os.path.join(project_dir, train_dir, 'pain')
    train_no_pain_dir = os.path.join(project_dir, train_dir, 'no-pain')
    val_pain_dir = os.path.join(project_dir, val_dir, 'pain')
    val_no_pain_dir = os.path.join(project_dir, val_dir, 'no-pain')
    
    os.makedirs(train_pain_dir, exist_ok=True)
    os.makedirs(train_no_pain_dir, exist_ok=True)
    os.makedirs(val_pain_dir, exist_ok=True)
    os.makedirs(val_no_pain_dir, exist_ok=True)

    # ensure the directories are empty
    for dir_path in [train_pain_dir, train_no_pain_dir, val_pain_dir, val_no_pain_dir]:
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    pain_images = os.listdir(pain_dir)
    no_pain_images = os.listdir(no_pain_dir)
    
    random.shuffle(pain_images)
    random.shuffle(no_pain_images)
    
    train_pain_images = pain_images[:int(0.8*len(pain_images))]
    val_pain_images = pain_images[int(0.8*len(pain_images)):]
    train_no_pain_images = no_pain_images[:int(0.8*len(no_pain_images))]
    val_no_pain_images = no_pain_images[int(0.8*len(no_pain_images)):]
    
    for image in train_pain_images:
        shutil.copy(os.path.join(pain_dir, image), os.path.join(train_pain_dir, image))
    for image in val_pain_images:
        shutil.copy(os.path.join(pain_dir, image), os.path.join(val_pain_dir, image))
    for image in train_no_pain_images:
        shutil.copy(os.path.join(no_pain_dir, image), os.path.join(train_no_pain_dir, image))
    for image in val_no_pain_images:
        shutil.copy(os.path.join(no_pain_dir, image), os.path.join(val_no_pain_dir, image))
    
    print("Split complete!")



def main():
    dataset_dir = r'C:\Users\dylan\OneDrive - University of Cambridge\Part II\Affective AI\Project'
    project_dir = r'C:\Users\dylan\work\AAI-Mini-Project\CNN-classifier'
    image_dir = 'UNBC/Images'
    label_dir = 'UNBC/Frame_Labels/PSPI'
    train_dir = 'data/train'
    val_dir = 'data/val'

    # Example usage of label_images
    # labels = label_images(label_dir, dataset_dir)
    # print(labels['ll042t1aaaff001.png']) # 0 = no pain (PSPI = 0)
    # print(labels['ll042t1aaaff022.png']) # 1 = pain (PSPI = 2)

    # uncomment to run sort_images
    # sort_images(image_dir, label_dir, project_dir, dataset_dir)

    # uncomment to run split_images
    split_images(project_dir, train_dir, val_dir)

if __name__ == '__main__':
    main()