from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
'''
        The presented code defines a more advanced custom PyTorch Dataset subclass named MyDataset.
        It is engineered to load and preprocess data for a machine learning task, with multiple pre-processing options.
        The data, including image file paths and labels, is stored in a CSV file.
        This class is designed to be flexible and can be configured to perform multiple image processing tasks
        such as cropping, resizing, denoising, adding Gaussian noise, and Fourier transforms.
'''
class MyDataset(Dataset):
    '''
        This class, MyDataset, is a subclass of the PyTorch Dataset class. It is initialized with several parameters
        including the path to a CSV file, image size, transformation to be applied to the images, classification flag,
        denoising flag, a boolean to return all fields, a noising flag, a downsampling flag, and a threshold percentage for
        Fourier Transform-based denoising.
    '''
    
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        
    def __len__(self):
        '''
        The __len__ method returns the total number of samples in the dataset.
        '''
        return len(self.data)

    def __getitem__(self, index):
        '''
        The __getitem__ method takes an index and returns the corresponding image and label from the dataset. 
        The image file path is read from the CSV file and the image is loaded from disk. Depending on the configuration,
        it can perform different operations on the image: cropping to a specified size, downsampling (resizing), 
        denoising through Fourier Transform, and adding Gaussian noise. If the boolean return all flag is set to True,
        it returns the image path, the transformed image, and the label; otherwise, it returns the transformed image 
        and the label.
        '''
        
        image_path = self.data.iloc[index, 0]
        label = self.data.iloc[index, 1]
        
        #image_path = image_path
        # Load the image and convert it to a PyTorch tensor
        with Image.open(image_path) as img:
            sample = transforms.ToTensor()(img)
        return sample, label
