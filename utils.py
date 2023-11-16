from sklearn.preprocessing import LabelEncoder
from models import regressor, classifier
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mydataset import MyDataset
import seaborn as sns
import pandas as pd 
import numpy as np 
import torch 
import json
import os

# Function to create a directory if it doesn't exist
def create_dir(path): 
    if not os.path.isdir(path):
        os.makedirs(path)
    return path

def create_model(activation_function, dropout_strength, model_type = 'regressor'):
    model = regressor.Regressor(activation_function, dropout_strength) if model_type == 'regressor' else classifier.Classifier(activation_function, dropout_strength) 
    return model

def log_transform(y):
    return torch.log(y + 1)

def inverse_log_transform(y):
    return torch.exp(y) - 1

def load_dataset(training_data_path, testing_data_path, validation_data_path, 
                 batch_size, model_type = 'regressor'):
    # Load the dataset
    training_dataset = MyDataset(training_data_path)
    testing_dataset = MyDataset(testing_data_path)
    validation_dataset = MyDataset(validation_data_path)
    
    
    training_length = len(training_dataset)
    testing_length = len(testing_dataset)
    validation_length = len(validation_dataset)
    
    print(f'Training length: {training_length} \nTesting length: {testing_length} \nValidation length: {validation_length}')
    iterations = training_length//batch_size

    label_encoder = None
    if model_type == 'classifier': 
        training_dataloader = DataLoader(training_dataset, batch_size = batch_size, shuffle = True)
        label_encoder = fit_label_encoder(training_dataloader) 
    return training_dataset, testing_dataset, validation_dataset, iterations, label_encoder

      
# This function plots the losses and saves them to a path
def plot_losses(training_losses, validation_losses, xlabel, ylabel, title, path_to_save):
    plt.figure()
    plt.plot(training_losses, label='Training loss')
    plt.plot(validation_losses, label='Validation loss')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(path_to_save+'/losses.png')
    plt.close()
        
def decode_categorical_labels(y, label_encoder):
    # Decode the categorical labels
    #TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
    y = y.cpu().numpy()
    y = label_encoder.inverse_transform(y)
    return y

def encode_categorical_labels(y, label_encoder):
    # Encode the categorical labels
    y = label_encoder.transform(y)
    return y

def fit_label_encoder(dataloader):
    # Encode the categorical labels
    label_encoder = LabelEncoder()
    for batch_data, batch_labels in dataloader: 
        label_encoder.fit(batch_labels)
    return label_encoder

def get_classes(label_encoder):
        classes = []
        for i, item in enumerate(label_encoder.classes_):
            classes.append(item)
        return classes

def plot_confusion_matrix(cm, classes, path, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(8,8))
    # Convert the type of the confusion matrix to int
    cm = cm.astype('int')
    # Plot the heatmap
    
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar_kws={'shrink': .82}, square=True, xticklabels=classes, yticklabels=classes)

    #make the font bigger
    sns.set(font_scale=1.2)
    #make the x-axis and y-axis labels bigger
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    #make the ylabel title bigger
    
    #Make the digitis inside the heatmap and the colorbar bigger
    cbar = plt.gcf().axes[-1]
    cbar.tick_params(labelsize=18)
    
    
    plt.title(title)
    plt.ylabel('True label', fontsize = 18)
    plt.xlabel('Predicted label', fontsize = 18)
    #plt.show()
    plt.savefig(path+'/confusion_matrix.png')


def plot_conv(conv, path):
    # Get the number of filters
    num_filters = conv.shape[0] # 32
    fig , ax = plt.subplots(4, 8, figsize=(20, 10))
    for i in range(num_filters):
        image_per_filter = conv[:, i, :, :]
        #print(image_per_filter.shape)
        image_per_filter = np.squeeze(image_per_filter)
        #print(image_per_filter.shape)
        ax[i//8, i%8].imshow(image_per_filter, cmap='gray')
        ax[i//8, i%8].axis('off')
    plt.savefig(path)
    plt.close()
    
def plot_convolutional_results(model, missclassified_test, new_path):
    # Create a figure
    fig, ax = plt.subplots(3, 4, figsize=(10, 10))
    # For each class
    for i, c in enumerate(missclassified_test):
        # Try to get an image
        try: 
            # Get the image
            image = missclassified_test[c]['image'][0]
            image = image.view(1, 1, 512, 512)
            model.eval()

            def hook_fn(module, input, output):
                activations.append(output)

            # Register hooks
            activations = []
            handle_conv1 = model.conv1.register_forward_hook(hook_fn)
            handle_conv2 = model.conv2.register_forward_hook(hook_fn)
            handle_conv3 = model.conv3.register_forward_hook(hook_fn)

            with torch.no_grad():
                model(image)

            # Remove hooks
            handle_conv1.remove()
            handle_conv2.remove()
            handle_conv3.remove()
            
            # Plot the output of the 1st convolutional layer        
            plot_conv(activations[0].detach().cpu().numpy(), new_path+f'/class_{c}conv1.png')
            plot_conv(activations[1].detach().cpu().numpy(), new_path+f'/class_{c}conv2.png')
            plot_conv(activations[2].detach().cpu().numpy(), new_path+f'/class_{c}conv3.png')
        except: 
            # There is no image for this class, so skip it
            continue
        ax[i, 0].imshow(image.squeeze().cpu().numpy(), cmap='gray')
        ax[i, 0].set_title(f"Real: {c},\nPredicted: {missclassified_test[c]['predicted_label'][0]}")
        ax[i, 0].axis('off')
        

        
        #conv_1_image = activations[0].detach().cpu().numpy()
        #conv_1_image = conv_1_image[0, :, :, :]
        # Shape [32, 508, 508]
        
        # Get the maximum value of the output of the 1st convolutional layer
        max_val, _ = torch.max(activations[0], dim=1)
        #print(f'Shape of max_val: {max_val.shape}') #[1, 508, 508]
        
        # The shape 
        ax[i, 1].imshow(max_val.squeeze().cpu(), cmap='gray')
        ax[i, 1].set_title('Output of\n1st conv layer')
        ax[i, 1].axis('off')
        
        # Plot the output of the 2nd convolutional layer
        #conv_2_image = activations[1].detach().cpu().numpy()
        #conv_2_image = conv_2_image[0, :, :, :]
        max_val, _ = torch.max(activations[1], dim=1)
        
        ax[i, 2].imshow(max_val.squeeze().cpu(), cmap='gray')
        ax[i, 2].set_title('Output of\n2nd conv layer')
        ax[i, 2].axis('off')
        
        # Plot the output of the 3rd convolutional layer
        #conv_3_image = activations[2].detach().cpu().numpy()
        #conv_3_image = conv_3_image[0, :, :, :]
        max_val, _ = torch.max(activations[2], dim=1)
        
        ax[i, 3].imshow(max_val.squeeze().cpu(), cmap='gray')
        ax[i, 3].set_title('Output of\n3rd conv layer')
        ax[i, 3].axis('off')
    # Save the figure
    # Give the name "convolutional_results.png"
    plt.savefig(new_path + '/convolutional_results.png')
    plt.close()