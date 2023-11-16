''' 
Regression Model for predicting values of the meterial properties of the images
'''
from utils import load_dataset, create_model, log_transform, inverse_log_transform, create_dir, plot_losses
from torch.utils.data import DataLoader
from torchsummary import summary
import torch.nn as nn
import pandas as pd
import argparse 
import torch 
      


def train_model(model, optimizer, loss_function, training_dataset, validation_dataset, epochs, batch_size, device, label_name):
    
    training_length = len(training_dataset)
    iterations = training_length//batch_size
    
    validation_losses = []
    training_losses = []
    
    for epoch in range(epochs):
        epoch_train_loss = 0
        epoch_validation_loss = 0
        # Reset the dataloader for each epoch
        training_dataloader = DataLoader(training_dataset, batch_size = batch_size, shuffle = True)
        for iter, i in enumerate(range(0, training_length, batch_size)): 
            print(f'Epoch: {epoch+1}/{epochs} \t Iteration: {iter+1}/{iterations}', end='\r')
            # Get new batch of data
            batch_X, batch_y = next(training_dataloader.__iter__()) 
            batch_y = batch_y.to(device)
            #RuntimeError: Found dtype Double but expected Float
            batch_y = batch_y.float()
            # make batch_y [batch_size, 1]
            batch_y = batch_y.reshape(-1, 1)

            ###############################################################
            # Transform the labels
            ###############################################################
            if label_name == 'np':
                batch_y = log_transform(batch_y)
            
            batch_X = (batch_X.float()/255).to(device)
            
            ###############################################################
            # Forward pass
            ###############################################################
            model.zero_grad()
            outputs = model(batch_X)
            loss = loss_function(outputs, batch_y)             
            
            ###############################################################
            # Backward pass
            ###############################################################
            loss.backward()
            epoch_train_loss += loss.item()
            optimizer.step()

        
        with torch.no_grad():
            validation_dataloader = DataLoader(validation_dataset, batch_size = batch_size, shuffle = True, pin_memory=True)
            validation_length = len(validation_dataset)
            for iter, i in enumerate(range(0, validation_length, batch_size)):
                val_batch_X, val_batch_y = next(validation_dataloader.__iter__())
                ###############################################################
                # Transform the labels
                ###############################################################
                val_batch_y = val_batch_y.to(device)
                val_batch_y = val_batch_y.reshape(-1, 1)
                if label_name == 'np':
                    val_batch_y = log_transform(val_batch_y)
                
                #val_batch_X = val_batch_X.unsqueeze(1)
                val_batch_X = (val_batch_X.float()/255).to(device)
                # Get the output
                val_outputs = model(val_batch_X)
                # Get the loss
                val_loss = loss_function(val_outputs, val_batch_y)
                # Add the loss to the epoch loss
                epoch_validation_loss += val_loss.item()
                # Delete unnecessary variables
                del val_batch_X, val_batch_y, val_outputs, val_loss
                
        # Get the average loss for the epoch
        training_loss = epoch_train_loss / (training_length//batch_size)
        training_losses.append(training_loss)
        validation_loss = epoch_validation_loss / (validation_length//batch_size)
        validation_losses.append(validation_loss)
        
    return model, training_losses, validation_losses


def test_model(model, testing_dataset,batch_size, device, label_name):
    iterations = len(testing_dataset)//batch_size
    testing_length = len(testing_dataset)
    # create a pandas dataframe to store the results
    # The dataframe will have 2 columns: 
    # 1. The predicted angle
    # 2. The actual angle
    # The rows will be the number of iterations
    # The dataframe will be saved as a csv file
    testing_results = pd.DataFrame(columns = ['actual_value', 'predicted_value'])
    testing_loss = 0
    test_i = 0
    testing_dataloader = DataLoader(testing_dataset, batch_size = batch_size, shuffle = True, pin_memory=True)
    with torch.no_grad():
        for iter, i in enumerate(range(0, testing_length, batch_size)):
            print(f'Iteration: {iter}/{iterations}', end='\r')
            batch_X, batch_y = next(testing_dataloader.__iter__())
            batch_y = batch_y.to(device)
            batch_y = batch_y.float()
            batch_y = batch_y.reshape(-1, 1)
            ###############################################################
            # Transform the labels
            ###############################################################
            if label_name == 'np':
                batch_y = log_transform(batch_y)

            #batch_X = batch_X.unsqueeze(1)
            batch_X = (batch_X.float()/255).to(device)
            outputs = model(batch_X)
            
            loss = loss_function(outputs, batch_y)
            # loss to double
            loss = loss.double()
            testing_loss += loss.item()
            
            
            ###############################################################
            # Inverse transform the outputs and the labels
            ###############################################################
            if label_name == 'np':
                outputs = inverse_log_transform(outputs)
                batch_y = inverse_log_transform(batch_y)
                
            # Add the results to the dataframe
            tmp = outputs.cpu().numpy()
            for j in range(tmp.shape[0]):
                testing_results.loc[test_i, 'predicted_value'] = outputs.cpu().numpy()[j]
                testing_results.loc[test_i, 'actual_value'] = batch_y.cpu().numpy()[j] 
                test_i += 1
                
    testing_loss = testing_loss / (testing_length//batch_size)
    
    return testing_results, testing_loss

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
    result_path = './Results/Regressor'
    create_dir(result_path)
    
    ###################################################################################
    # Arguments parser
    ###################################################################################
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-b', '--batch_size', type=int, default=32)
    argparser.add_argument('-e', '--epochs', type=int, default=200)
    argparser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
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


        
    if label_name == 'Angle' or label_name == 'angle' or label_name == 'a': 
        label_name ='angle'
    elif label_name == 'NP' or label_name == 'np' or label_name == 'n': 
        label_name = 'np'
    elif label_name == 'EP1' or label_name == 'ep1' or label_name == 'e':
        label_name = 'ep1'        
    else: 
        raise ValueError('Please select a valid label name to train on (Angle, NP, EP1)')
    ###################################################################################
    
    loss_functions = {'MSE': nn.MSELoss()}    

    loss_function = loss_functions[loss_func]
       
    experiment_string = f'batch_size_{batch_size}|epochs_{epochs}|learning_rate_{learning_rate}|dropout_strength_{dropout_strength}|label_{label_name}'

    ############################################################

    c_v = 1 if cross_val_bool == False else 5

    c_v_predictions_and_actual_list = []
    c_v_loss_list = []
    for i in range(c_v):
        
        validation_path = f'{result_path}/{experiment_string}/cross_val_{i+1}'
        create_dir(validation_path)
        ######################################
        # CROSS VALIDATION DATASETS
        ######################################
        training_data_path = f'./datasets/cross_validation_uniform_data_{i+1}/{label_name}/train.csv'
        testing_data_path = f'./datasets/cross_validation_uniform_data_{i+1}/{label_name}/test.csv'
        validation_data_path = f'./datasets/cross_validation_uniform_data_{i+1}/{label_name}/valid.csv'
            
        print(f'Epochs: {epochs}, Learning Rate: {learning_rate}, Dropout strength: {dropout_strength} ,activation function: {activation_func} \n')
        
        # Create the model
        model = create_model(activation_func, dropout_strength).to(device)
        
        # Choose optimizer and loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)#.to(device)
        
        # Load the dataset
        training_dataset, testing_dataset, validation_dataset, iterations, _  = load_dataset(training_data_path, testing_data_path, validation_data_path, 
                                                                                                                batch_size)
        # Train the model                                 
        model, training_loss, validation_loss = train_model(model, optimizer, loss_function, training_dataset, validation_dataset, epochs, batch_size, device, label_name)
        
        # Save the model
        torch.save(model.state_dict(), validation_path+'/model.pt')
        
        # Test the model
        testing_results, testing_loss = test_model(model, testing_dataset,batch_size, device, label_name) 
        
        c_v_predictions_and_actual_list.append(testing_results)
        c_v_loss_list.append(testing_loss)
        # Plot the loss
        plot_losses(training_loss, validation_loss, xlabel = 'Epoch', ylabel = 'Mean Squared Error', title = 'Regression Loss', path_to_save=validation_path)
        
        print(f'\nCross validation {i+1} testing loss: {testing_loss}')
    
    if c_v != 1: 
        mean_testing_loss = sum(c_v_loss_list) / len(c_v_loss_list)
        print(f'Mean testing loss: {mean_testing_loss}')
        
