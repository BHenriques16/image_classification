# library import
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import random


# list of the different folders
folders = ['dataset_waste_container/container_battery', 
           'dataset_waste_container/container_biodegradable', 
           'dataset_waste_container/container_blue',
           'dataset_waste_container/container_default',
           'dataset_waste_container/container_green',
           'dataset_waste_container/container_oil',
           'dataset_waste_container/container_yellow'
            ]  

# lighting variability
for folder in folders:
    images = os.listdir(folder)
    mean_brightness = []

    for img_name in images:
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)                      # reads image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # converts to shades of gray 
        mean_brightness.append(gray.mean())             # calculates and saves the average brightness per image             
    #print(f'{folder.split("/")[-1]} - Mean: {np.mean(mean_brightness):.2f} | Standard Deviation: {np.std(mean_brightness):.2f}')


# angle problem
for folder in folders:
    images = os.listdir(folder)
    samples = random.sample(images, 5)  # selects 5 random images

    fig, axs = plt.subplots(1, 5, figsize=(15, 6)) 
    fig.suptitle(f'Amostras: {folder.split("/")[-1]}', fontsize=16)

    for i, img_name in enumerate(samples):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axs[i].imshow(img_rgb)
        axs[i].axis('off')
        axs[i].set_title(img_name)

    plt.tight_layout()
    #plt.show()


# similar colors between classes
mean_colors = {}

for folder in folders:
    images = os.listdir(folder)
    sample_images = images[:20]
    colors = []
    for img_name in sample_images:
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)          # convert to RGB
            avg_color_per_row = np.average(img_rgb, axis=0)         # mean per row
            avg_color = np.average(avg_color_per_row, axis=0)       # total mean of an image
            colors.append(avg_color)
    mean_colors[folder.split('/')[-1]] = np.mean(colors, axis=0)

print('Mean RGB colors per class:')
for classe, color in mean_colors.items():
    print(f'{classe}: R={color[0]:.1f}, G={color[1]:.1f}, B={color[2]:.1f}')
print('\n')    

            
# Number of images per class
data_dir = 'dataset_waste_container'

class_counts = {}

for class_folder in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, class_folder)
    if os.path.isdir(folder_path):
        num_images = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
        class_counts[class_folder] = num_images

for classe, total in class_counts.items():
    print(f'{classe}: {total} imagens')