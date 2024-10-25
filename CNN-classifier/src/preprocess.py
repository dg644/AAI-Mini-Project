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
                    labels[filename] = 1
                else:
                    labels[filename] = 0
    
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
            dest_dir = os.path.join(project_dir, 'data', 'sorted-images', 'pain')
        else:
            dest_dir = os.path.join(project_dir, 'data', 'sorted-images', 'no-pain')
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, filename)
        if not os.path.exists(dest_path):
            shutil.copy(image_path, dest_path)

dataset_dir = r'C:\Users\dylan\OneDrive - University of Cambridge\Part II\Affective AI\Project'
project_dir = r'C:\Users\dylan\work\AAI-Mini-Project\CNN-classifier'
image_dir = 'UNBC/Images'
label_dir = 'UNBC/Frame_Labels/PSPI'
# Example usage of label_images
# labels = label_images(label_dir, dataset_dir)
# print(labels['ll042t1aaaff001.png']) # 0 = no pain (PSPI = 0)
# print(labels['ll042t1aaaff022.png']) # 1 = pain (PSPI = 2)
sort_images(image_dir, label_dir, project_dir, dataset_dir)


