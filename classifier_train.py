''' 
Classifier Model for predicting the type of laser hit on the material
'''
from utils import load_dataset, create_model, create_dir, decode_categorical_labels, encode_categorical_labels, get_classes
from utils import plot_losses, plot_confusion_matrix, plot_convolutional_results
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchsummary import summary
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np     
import argparse 
import torch 
import os                    
                 

def train_model(model, optimizer, loss_function, training_dataset, validation_dataset, epochs, batch_size, device, label_encoder):
    '''
    Trains the classifier model on the given dataset.

    Args:
        model: The neural network model to be trained.
        optimizer: The optimization algorithm.
        loss_function: The loss function to be used for training.
        training_dataset: Dataset for training the model.
        validation_dataset: Dataset for validating the model.
        epochs: Number of training epochs.
        batch_size: Size of each data batch.
        device: Device to perform computations on (CPU/GPU).
        label_encoder: Encoder for converting labels to and from categorical format.

    Returns:
        model: The trained model.
        training_losses: List of training losses per epoch.
        validation_losses: List of validation losses per epoch.
    '''
    
    # Initializing variables for training
    training_length = len(training_dataset)
    iterations = training_length//batch_size
    validation_losses = []
    training_losses = []
    
    # Training loop for each epoch
    for epoch in range(epochs):
        epoch_train_loss = 0
        epoch_validation_loss = 0
        training_dataloader = DataLoader(training_dataset, batch_size = batch_size, shuffle = True)
        
        # Iterating through the training dataset
        for iter, i in enumerate(range(0, training_length, batch_size)):
            print(f'Epoch: {epoch}, iteration: {iter}/{iterations}', end='\r')

            # Fetching the batch
            batch_X, batch_y = next(training_dataloader.__iter__()) 

            # Encoding categorical labels for the batch            
            batch_y = encode_categorical_labels(list(batch_y), label_encoder)
            batch_y = torch.from_numpy(batch_y).long()
            batch_y = torch.nn.functional.one_hot(batch_y, num_classes=3).to(device)
            
            # Normalizing and transferring batch to device
            batch_X = (batch_X.float()/255).to(device)

            # Forward pass: Computing model outputs and loss
            model.zero_grad()
            outputs = model(batch_X)
            loss = loss_function(outputs, torch.max(batch_y, 1)[1])       
            
            # Backward pass: Computing gradients and updating weights
            loss.backward()
            epoch_train_loss += loss.item()
            optimizer.step()
            

        # Validation phase, no gradient calculations needed
        with torch.no_grad():
            validation_dataloader = DataLoader(validation_dataset, batch_size = batch_size, shuffle = True, pin_memory=True)            
            validation_length = len(validation_dataset)
            
            # Iterating through the validation dataset
            for iter, i in enumerate(range(0, validation_length, batch_size)):
                val_batch_X, val_batch_y = next(validation_dataloader.__iter__())
                
                # Encoding categorical labels for validation batch
                val_batch_y = encode_categorical_labels(list(val_batch_y), label_encoder)
                val_batch_y = torch.from_numpy(val_batch_y).long()
                val_batch_y = torch.nn.functional.one_hot(val_batch_y, num_classes=3).to(device)
                
                # Normalizing and transferring validation batch to device
                val_batch_X = (val_batch_X.float()/255).to(device)
                
                # Model inference on validation batch
                val_outputs = model(val_batch_X)
                val_loss = loss_function(val_outputs, torch.max(val_batch_y, 1)[1])
                
                # Accumulating validation loss
                epoch_validation_loss += val_loss.item()
            
        # Calculating and storing average losses for the epoch
        training_loss = epoch_train_loss / (training_length//batch_size)
        training_losses.append(training_loss)
        validation_loss = epoch_validation_loss / (validation_length//batch_size)
        validation_losses.append(validation_loss)

    return model, training_losses, validation_losses

def test_model(model, testing_dataset, batch_size, device, label_encoder):
    '''
    Tests the trained classifier model on the testing dataset.

    Args:
        model: The trained neural network model.
        testing_dataset: Dataset for testing the model.
        batch_size: Size of each data batch.
        device: Device to perform computations on (CPU/GPU).
        label_encoder: Encoder for converting labels to and from categorical format.

    Returns:
        accuracy_test: Accuracy of the model on the testing dataset.
        missclassified_test: Details of misclassified instances.
        confusion_mtx: Confusion matrix of the model predictions.
        prediction_types: Predicted labels.
        real_types: Actual labels.
    '''
    # Initialize variables for testing
    iterations = len(testing_dataset)//batch_size
    testing_length = len(testing_dataset)
    
    # Retrieve class labels
    classes = get_classes(label_encoder)
    missclassified_test = {c: {'predicted_label': [], 'image': []} for c in classes}
    counted_classes = {c: 0 for c in classes}

    # Initialize confusion matrix
    confusion_mtx = np.zeros((3, 3)) 
    
    # Lists for storing predictions and actual labels
    prediction_types = []
    real_types = []
    
    # Variables for tracking accuracy
    correct = 0
    total = 0
    
    # DataLoader for testing
    testing_dataloader = DataLoader(testing_dataset, batch_size = batch_size, shuffle = True, pin_memory=True)
    
    # Begin testing - no gradient calculations
    with torch.no_grad():
        for iter, i in enumerate(range(0, testing_length, batch_size)):
            print(f'Iteration: {iter}/{iterations}', end='\r')
            
            # Get the current batch
            batch_X, batch_yy = next(testing_dataloader.__iter__())
            
            # Encode labels to categorical format
            batch_y = encode_categorical_labels(list(batch_yy), label_encoder)
            batch_y = torch.from_numpy(batch_y).long()
            batch_y = torch.nn.functional.one_hot(batch_y, num_classes=3).to(device)
            
            # Normalize and transfer batch to device
            batch_X = (batch_X.float()/255).to(device)
            
            # Model inference
            outputs = model(batch_X)
           
            # Determine predicted and actual classes
            predicted = torch.max(outputs.data, 1)[1] 
            real = torch.max(batch_y, 1)[1] 
            
            # Update confusion matrix
            real_np = real.cpu().numpy()
            predicted_np = predicted.cpu().numpy()
            confusion_mtx += confusion_matrix(real_np, predicted_np, labels=np.arange(3))
            
            # Update correct and total counts
            correct += (predicted == real).sum().item()
            total += batch_y.size(0)            

            # Decode the labels to original format
            dec_batch_y = decode_categorical_labels(real, label_encoder)
            dec_predicted = decode_categorical_labels(predicted, label_encoder)

            # Store predictions and actual labels
            prediction = np.array(dec_predicted).flatten()
            prediction_types.append(prediction)
            batch_y = np.array(list(batch_yy)).flatten()
            real_types.append(batch_yy)
            
            # Track misclassified cases
            for i, (p, r) in enumerate(zip(dec_predicted, dec_batch_y)):
                counted_classes[r] += 1
                if p != r:
                    missclassified_test[r]['predicted_label'].append(p)
                    missclassified_test[r]['image'].append(batch_X[i])  
    
    # Calculate final accuracy         
    accuracy_test = round(correct/total, 3)
    
    return accuracy_test, missclassified_test, confusion_mtx, prediction_types, real_types



if __name__ == '__main__':
    # Main script execution
    # Includes argument parsing, model initialization, training, testing, and result visualization
    if torch.cuda.is_available():
        # Use GPU
        print("Using GPU")
        device = torch.device("cuda")
    else:
        # Use CPU
        print("Using CPU")
        device = torch.device("cpu")

    # Create a results folder if it does not exist
    result_path = './Results/Classifier'
    create_dir(result_path)
        
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-b', '--batch_size', type=int, default=32)
    argparser.add_argument('-e', '--epochs', type=int, default=200)
    argparser.add_argument('-lr', '--learning_rate', type=float, default=0.0005)
    argparser.add_argument('-d','--dropout_strength', type=float, default=0.2)
    argparser.add_argument('-l', '--label', type=str, default='Angle')
    argparser.add_argument('--loss_function', type = str, default = 'MSE')
    argparser.add_argument('--activation_function', type = str, default = 'relu')
    argparser.add_argument('-c', '--cross_val', type = bool, default = False)

    args = argparser.parse_args()
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    dropout_strength = args.dropout_strength
    label_name = args.label
    loss_func = args.loss_function
    activation_func = args.activation_function
    cross_val_bool = args.cross_val


    # Loss Function is the Cross Entropy Loss for 3 classes
    loss_function = nn.CrossEntropyLoss()
    experiment_string = f'batch_size_{batch_size}|epochs_{epochs}|learning_rate_{learning_rate}|dropout_strength_{dropout_strength}'

    c_v = 1 if cross_val_bool == False else 5

    c_v_accuracy_list = []
    for i in range(c_v):

        validation_path = f'{result_path}/{experiment_string}/cross_Val_{i+1}'
        create_dir(validation_path)

        training_data_path = f'./datasets/cross_validation_uniform_data_{i+1}/pp1/train.csv'
        testing_data_path = f'./datasets/cross_validation_uniform_data_{i+1}/pp1/test.csv'
        validation_data_path = f'./datasets/cross_validation_uniform_data_{i+1}/pp1/valid.csv'
        
        #Create the model
        model = create_model(activation_func, dropout_strength, model_type = 'classifier').to(device)
        
        # Choose optimizer and loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Load the dataset
        training_dataset, testing_dataset, validation_dataset, iterations, label_encoder  = load_dataset(training_data_path, testing_data_path, validation_data_path, 
                                                                                                            batch_size, model_type = 'classifier')
        # Train the model
        model, training_loss, validation_loss = train_model(model, optimizer, loss_function, training_dataset, validation_dataset, epochs, batch_size, device, label_encoder)
        
        # Save the model
        torch.save(model.state_dict(), validation_path +'/model.pt')
        
        # Test the model
        accuracy_test, missclassified_test, confusion_mtx, prediction_types, real_types = test_model(model, testing_dataset, batch_size, device, label_encoder)
        c_v_accuracy_list.append(accuracy_test)
        
        # Plot the loss
        plot_losses(training_loss, validation_loss, xlabel = 'Epoch', ylabel = 'Cross Entropy', title = 'Classification Loss', path_to_save = validation_path)
        
        # Plot the convolutional results
        plot_convolutional_results(model, missclassified_test, validation_path)

        # Plot the confusion matrix
        classes = list(label_encoder.classes_)
        plot_confusion_matrix(confusion_mtx, classes=classes, title='Confusion Matrix', cmap=plt.cm.Blues, path=validation_path)
        
        print(f'\nCross validation {i+1} testing accuracy: {accuracy_test}')
        
    if c_v != 1: 
        mean_testing_loss = sum(c_v_accuracy_list) / len(c_v_accuracy_list)
        print(f'Mean accuracy: {mean_testing_loss}')
        

    
    