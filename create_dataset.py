from utils import create_dir
import pandas as pd 
import random
import cv2
import os 
import re


def create_dataset(in_dirs, out_dir_images, height, width, desired_dim, path):
    '''
    Generates a dataset by cropping and saving portions from images in specified directories.

    Args:
        in_dirs (list): List of input directories containing images.
        out_dir_images (str): Path to the output directory where processed images will be saved.
        height (int): Height of the original images.
        width (int): Width of the original images.
        desired_dim (int): Desired dimension for each cropped image (assumes square crop).
        path (str): Base path for the input directories.

    Returns:
        None: The function saves cropped images to the specified output directory.
    '''
    c=1
    for _dir in in_dirs:
        file=os.listdir(path+'/'+_dir)
        # Replace '/' with '_' in directory names to store in a new directory
        replaced_dir = _dir.replace('/', '_') if 'Paper Data' in _dir else _dir
        for im in file:
            if '.png' in im:
                # Create a new directory for each image
                tmp_dir_image = create_dir(out_dir_images+'/'+ str(c))
                
                # Read the image in grayscale
                image = cv2.imread(os.path.join(path+'/'+_dir,im),cv2.IMREAD_GRAYSCALE)
                
                # Crop and save parts of the image
                for n in range(number_of_batches):
                    x_cord = random.randint(0, height-desired_dim)
                    y_cord = random.randint(0, width-desired_dim)
                    batch_image = image[x_cord:x_cord+desired_dim, y_cord:y_cord+desired_dim]
                    batch_name = replaced_dir+'_'+im[:-4]+'_'+str(n+1)+'.bmp'
                    cv2.imwrite(tmp_dir_image+'/'+batch_name, batch_image)
        
            c+=1


def create_data_with_labels_csv(data_xlsx, images_path, path_to_csv_data):
    '''
    Creates labeled datasets in CSV format for training, validation, and testing from provided image data.

    Args:
        data_xlsx (DataFrame): DataFrame containing image names and associated labels.
        images_path (str): Directory path where the images are stored.
        path_to_csv_data (str): Directory path to save the generated CSV files.

    Returns:
        None: The function generates CSV files containing image paths and labels for each dataset type.
    '''
    data = os.listdir(images_path)
    # Split data into training, validation, and testing sets 
    split_idx_1 = int(len(data) * 0.8)
    split_idx_2 = int(len(data) * 0.9)
    training_data = data[:split_idx_1]
    validation_data = data[split_idx_1:split_idx_2]
    testing_data = data[split_idx_2:]
 
    # Helper function to process and label data
    def process_data(data_subset, dataset_type):
        data_with_labels = []
        for dir_num in data_subset:
            new_path = os.path.join(images_path, dir_num)
            for image in os.listdir(new_path):
                image_name = re.sub(r'_\d+.bmp', '', image)
                image_name = image_name.replace("_", "/") + ".bmp"

                # Extract labels from the xlsx file
                angle = data_xlsx.loc[data_xlsx['Names'] == image_name, 'angle [deg]'].iloc[0]
                PP1 = data_xlsx.loc[data_xlsx['Names'] == image_name, 'PP1'].iloc[0]
                PP1 = data_xlsx.loc[data_xlsx['Names'] == image_name, 'PP2'].iloc[0] if PP1 in ['0', 0] else PP1
                NP = data_xlsx.loc[data_xlsx['Names'] == image_name, 'NP'].iloc[0]
                EP1 = data_xlsx.loc[data_xlsx['Names'] == image_name, 'EP1 [μJ]'].iloc[0]
                
                image_path = os.path.join(new_path, image)
                data_with_labels.append({'image_path': image_path, 'angle': angle, 'PP1': PP1, 'NP': NP, 'EP1': EP1})

        # Save to CSV
        df = pd.DataFrame(data_with_labels)
        df.to_csv(os.path.join(path_to_csv_data, f'{dataset_type}_data.csv'), index=False)
        
    # Create directories for CSV data
    create_dir(path_to_csv_data)

    # Process and save each data subset
    process_data(training_data, 'training')
    process_data(validation_data, 'validation')
    process_data(testing_data, 'testing')


if __name__ == '__main__':
    path = os.getcwd()
    
    # Dimensions of the original and desired cropped images    
    height = 960
    width = 1280
    number_of_batches = 100
    desired_dim = 256


    
    # Directories containing the original image data
    in_dirs =[
	'2020-4-30 tuning ripple period',
	'2020-6-9 Crossed polarized',
	'Paper Data/Double pulses',
	'Paper Data/Repetition 6p & 2p 29-4-2020',
	'Paper Data/Single pulses 2p',
	 'Paper Data/Single pulses 4 and half 6',
	'Paper Data/Repetition 6p & 2p 29-4-2020/Details'
    ]
    
    # Create the dataset and labeled data
    images_path = f'./datasets/2023_im_dataset_{desired_dim}x{desired_dim}'
    out_dir_images = create_dir(images_path)
    create_dataset(in_dirs, out_dir_images, height, width, desired_dim, path = './images')
    data_xlsx = pd.read_excel("./images/all_images.xlsx", engine = 'openpyxl')
    create_data_with_labels_csv(data_xlsx, images_path, path_to_csv_data = './datasets/data_with_labels_csv')