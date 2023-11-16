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
    training_length = len(training_dataset)

    iterations = training_length//batch_size
    validation_losses = []
    training_losses = []
    for epoch in range(epochs):
        epoch_train_loss = 0
        epoch_validation_loss = 0
        training_dataloader = DataLoader(training_dataset, batch_size = batch_size, shuffle = True)
        for iter, i in enumerate(range(0, training_length, batch_size)):
            # print in the same line
            print(f'Epoch: {epoch}, iteration: {iter}/{iterations}', end='\r')

            # Get the batch
            batch_X, batch_y = next(training_dataloader.__iter__()) 

            ###############################################################
            # Encode the categorical labels
            ###############################################################
            
            batch_y = encode_categorical_labels(list(batch_y), label_encoder)
            
            batch_y = torch.from_numpy(batch_y).long()
            batch_y = torch.nn.functional.one_hot(batch_y, num_classes=3).to(device)
            
            
            # print(batch_y)

            batch_X = (batch_X.float()/255).to(device)

            ###############################################################
            # Forward pass
            ###############################################################
            model.zero_grad()
            outputs = model(batch_X)
            loss = loss_function(outputs, torch.max(batch_y, 1)[1])       
            # print(f'outputs: {outputs}')
            # predicted= torch.max(outputs.data, 1)[1]
            # print(f'Predicted: {predicted}')
            # print(f'Decoded: {decode_categorical_labels(predicted, label_encoder)}')  
            
            # reversed_batch_y = torch.max(batch_y, 1)[1]
            # predicted_reversed = decode_categorical_labels(reversed_batch_y, label_encoder)
            # print(f'Predicted reversed: {reversed_batch_y}')
            # print(f'Decoded reversed: {predicted_reversed}')
            # sys.exit()
            ###############################################################
            # Backward pass
            ###############################################################
            loss.backward()
            epoch_train_loss += loss.item()
            optimizer.step()
            
            # Print first convolutional layer weights grads
            
            # Delete unnecessary variables
            del batch_X, batch_y, outputs, loss
            
            # Use the validation set to check the accuracy without updating the weights

        # No training
        with torch.no_grad():
            validation_dataloader = DataLoader(validation_dataset, batch_size = batch_size, shuffle = True, pin_memory=True)            
            validation_length = len(validation_dataset)
            for iter, i in enumerate(range(0, validation_length, batch_size)):
                val_batch_X, val_batch_y = next(validation_dataloader.__iter__())
                ###############################################################
                # Encode the categorical labels
                ###############################################################
                val_batch_y = encode_categorical_labels(list(val_batch_y), label_encoder)
                val_batch_y = torch.from_numpy(val_batch_y).long()
                val_batch_y = torch.nn.functional.one_hot(val_batch_y, num_classes=3).to(device)
                
                val_batch_X = (val_batch_X.float()/255).to(device)
                # Get the output
                val_outputs = model(val_batch_X)
                # Get the loss
                val_loss = loss_function(val_outputs, torch.max(val_batch_y, 1)[1])
                # Add the loss to the epoch loss
                epoch_validation_loss += val_loss.item()
                # Delete unnecessary variables
                del val_batch_X, val_batch_y, val_outputs, val_loss
            
                
            
        # Get the average loss for the epoch
        training_loss = epoch_train_loss / (training_length//batch_size)
        training_losses.append(training_loss)
        validation_loss = epoch_validation_loss / (validation_length//batch_size)
        validation_losses.append(validation_loss)
        #print(f"Epoch: {epoch}. Loss: {training_loss}. Validation loss: {validation_loss}")
    return model, training_losses, validation_losses

def test_model(model,  training_dataset, testing_dataset, validation_dataset,batch_size, device, label_encoder):
    iterations = len(testing_dataset)//batch_size
    testing_length = len(testing_dataset)

    classes = get_classes(label_encoder)
    missclassified_test = {c: {'predicted_label': [], 'image': []} for c in classes}
    counted_classes = {c: 0 for c in classes}
    #missclassified_train = {c: {'predicted_label': None, 'image': None} for c in classes}
    # Test the model
    
    confusion_mtx = np.zeros((3, 3)) 
    
    prediction_types = []
    real_types = []
    
    correct = 0
    total = 0
    testing_dataloader = DataLoader(testing_dataset, batch_size = batch_size, shuffle = True, pin_memory=True)
    with torch.no_grad():
        for iter, i in enumerate(range(0, testing_length, batch_size)):
            # print in the same line
            print(f'Iteration: {iter}/{iterations}', end='\r')
            #print(f'Iteration: {iter}/{len(testing_data)//batch_size}', end='\r')
            # Get the batch
            batch_X, batch_yy = next(testing_dataloader.__iter__())
            ###############################################################
            # Encode the categorical labels
            ###############################################################
            #batch_y = batch_y.to(device)
            batch_y = encode_categorical_labels(list(batch_yy), label_encoder)
            batch_y = torch.from_numpy(batch_y).long()
            batch_y = torch.nn.functional.one_hot(batch_y, num_classes=3).to(device)
            
            
            batch_X = (batch_X.float()/255).to(device)
            # Get the output
            outputs = model(batch_X)
            # Get the predicted class
            predicted = torch.max(outputs.data, 1)[1] #torch.argmax(testing_labels[i])
            # Get the real class
            real = torch.max(batch_y, 1)[1] #model(testing_data[i].view(-1, 1, 512, 512))[0]
            
            ##############################################################
            # UPDATE CONFUSION MATRIX
            real_np = real.cpu().numpy()
            predicted_np = predicted.cpu().numpy()
            confusion_mtx += confusion_matrix(real_np, predicted_np, labels=np.arange(3))
            ##############################################################
            
            # Add the number of correct predictions
            correct += (predicted == real).sum().item()
            # Add the total number of predictions
            total += batch_y.size(0)            

            # Decode the labels
            dec_batch_y = decode_categorical_labels(real, label_encoder)
            dec_predicted = decode_categorical_labels(predicted, label_encoder)

            prediction = np.array(dec_predicted).flatten()
            prediction_types.append(prediction)
            
            batch_y = np.array(list(batch_yy)).flatten()
            real_types.append(batch_yy)
            
            for i, (p, r) in enumerate(zip(dec_predicted, dec_batch_y)):
                # Count how many classes r are in the test set
                counted_classes[r] += 1
                if p != r:
                    missclassified_test[r]['predicted_label'].append(p)
                    missclassified_test[r]['image'].append(batch_X[i])  
            
    accuracy_test = round(correct/total, 3)
    
    return accuracy_test, missclassified_test, confusion_mtx, prediction_types, real_types



if __name__ == '__main__':
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
        accuracy_test, missclassified_test, confusion_mtx, prediction_types, real_types = test_model(model, training_dataset, testing_dataset, validation_dataset, batch_size, device, label_encoder)
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
        

    
    