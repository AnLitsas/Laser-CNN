from utils import create_dir
import pandas as pd 
import os 
import re

def uniform_labels(df_all_images, label_counts, label_coloumn):
    '''
    Splits the data into uniform training, validation, and test sets based on label distribution.

    Args:
        df_all_images (DataFrame): DataFrame containing all image data.
        label_counts (Series): Counts of each label in the dataset.
        label_column (str): The name of the column in df_all_images that contains the labels.

    Returns:
        tuple: Three DataFrames corresponding to training, validation, and test sets.
    '''
    training_set = []
    validation_set = []
    test_set = []

    # Calculate the sizes of training, validation, and test datasets
    total_data = df_all_images.shape[0]
    training_size = int(0.8 * total_data)
    validation_size = int(0.1 * total_data)
    test_size = int(0.1 * total_data)
    
    for label in label_counts.index: 
        # Filter the dataset for each label
        label_data = df_all_images[df_all_images[label_coloumn] == label]

        # Determine proportional count for each label in each dataset
        label_count = label_counts[label] 
        training_label_count = int(training_size * label_count / total_data)
        validation_label_count = int(validation_size * label_count / total_data)
        test_label_count = int(test_size * label_count / total_data)
        
        # Randomly sample data for each set
        training_label_data = label_data.sample(n = training_label_count)
        validation_label_data = label_data.drop(training_label_data.index).sample(n = validation_label_count)
        test_label_data = label_data.drop(training_label_data.index).drop(validation_label_data.index).sample(n = test_label_count)
        
        remaining_label_data = label_data.drop(training_label_data.index).drop(validation_label_data.index).drop(test_label_data.index)
        
        # Reset index for each subset
        training_label_data = training_label_data.reset_index(drop = True)
        validation_label_data = validation_label_data.reset_index(drop = True)
        test_label_data = test_label_data.reset_index(drop = True)
    
        # Distribute any remaining data evenly across the datasets
        count = 0
        for index, _ in remaining_label_data.iterrows():
            row = df_all_images.iloc[[index]]
            
            if count % 3 == 0:
                    training_label_data = pd.concat([training_label_data, row], axis = 0).reset_index(drop=True)
                    
            elif count % 3 == 1:

                    validation_label_data = pd.concat([validation_label_data, row], axis = 0).reset_index(drop=True)
            else:
                    test_label_data = pd.concat([test_label_data, row], axis = 0).reset_index(drop=True)
            count += 1
        
        # Append each label's data to the respective dataset
        training_set.append(training_label_data)
        validation_set.append(validation_label_data)
        test_set.append(test_label_data)
        
        # Combine data from all labels into final datasets
    train_pd = pd.concat(training_set, axis = 0).reset_index(drop=True)
    valid_pd = pd.concat(validation_set, axis = 0).reset_index(drop=True)
    test_pd = pd.concat(test_set, axis = 0).reset_index(drop=True)
    
    return train_pd, valid_pd, test_pd
    
def create_csv(pd_set, dataset, path_to_save, label):
    '''
    Creates a CSV file from a given dataset DataFrame with specific label column.

    Args:
        pd_set (DataFrame): The DataFrame to be converted into a CSV file.
        dataset (DataFrame): The original dataset used for reference.
        path_to_save (str): The path where the CSV file will be saved.
        label (str): The label column name that will be used in the new DataFrame.

    Returns:
        None: The function saves the DataFrame as a CSV file to the specified path.
    '''
    # Define the column names to be excluded from the final CSV
    column_names = ['angle', 'PP1', 'NP', 'EP1']
    column_names.remove(label)
    
    print(f'Creating csv for {path_to_save}')
    new_df = pd.DataFrame(columns = ['image_path', 'label'])
    
    # Iterate over each row in the DataFrame to be converted
    for index, row in pd_set.iterrows():
        name = row['Names']
        
        # Process the image name to match with the dataset
        name = name[:-4]  # Remove the '.bmp' extension
        name = name.replace('/', '_')  # Replace '/' with '_'

        # Compile a regex pattern to match image paths in the dataset
        pattern = re.compile(f".*{name}_.*$") 
        match_rows = dataset[dataset['image_path'].str.match(pattern)]
        matches = match_rows.shape[0]
        
        # Handle cases where the name might have additional characters like whitespace and '^'
        if matches < 100:
            name = name[:-2]
            pattern2 = re.compile(f".*{name}\s\^_.*$")
            match_rows = dataset[dataset['image_path'].str.match(pattern2)]
            matches = match_rows.shape[0]
            
        # Log if there are an unexpectedly high number of matches    
        if matches>100: 
            print(f'{name} has {matches} matches')

        # Drop the columns not needed for the final CSV
        match_rows = match_rows.drop(column_names, axis=1)
        # Rename the label column as 'label'
        match_rows = match_rows.rename(columns={label: 'label'})

        # Append the processed rows to the new DataFrame
        new_df = pd.concat([new_df, match_rows], axis=0).reset_index(drop=True)

    # Save the new DataFrame as a CSV file
    new_df.to_csv(path_to_save, index = False)

if __name__ == '__main__':
    # Load CSV files containing testing, training, and validation data
    dataset1 = pd.read_csv("./datasets/data_with_labels_csv/testing_data.csv")
    dataset2 = pd.read_csv("./datasets/data_with_labels_csv/training_data.csv")
    dataset3 = pd.read_csv("./datasets/data_with_labels_csv/validation_data.csv")
    
    # Combine the datasets into a single DataFrame
    dataset = pd.concat([dataset1, dataset2, dataset3])
    
    # Load the Excel file containing all images data
    df_all_images = pd.read_excel("./images/all_images.xlsx", engine = 'openpyxl')
    
    # Count the occurrences of each angle in the dataset
    angle_counts = df_all_images['angle [deg]'].value_counts()
    angle_coloumn = 'angle [deg]'
    # Split the data into uniform training, validation, and testing sets based on angle distribution
    angle_train_pd, angle_valid_pd, angle_test_pd = uniform_labels(df_all_images, angle_counts, angle_coloumn)

    # Repeat the process for other labels: PP1, NP, and EP1
    pp1_counts = df_all_images['PP1'].value_counts()
    pp1_coloumn = 'PP1'
    pp1_train_pd, pp1_valid_pd, pp1_test_pd = uniform_labels(df_all_images, pp1_counts, pp1_coloumn)

    np_counts = df_all_images['NP'].value_counts()
    np_coloumn = 'NP'
    np_train_pd, np_valid_pd, np_test_pd = uniform_labels(df_all_images, np_counts, np_coloumn)

    ep1_counts = df_all_images['EP1 [μJ]'].value_counts()
    ep1_coloumn = 'EP1 [μJ]'
    ep1_train_pd, ep1_valid_pd, ep1_test_pd = uniform_labels(df_all_images, ep1_counts, ep1_coloumn)
    
    # Print the lengths of the original training, testing, and validation datasets
    print(f'Original len: {len(angle_train_pd)}')
    print(f'Original test len: {len(angle_test_pd)}')
    print(f'Original valid len: {len(angle_valid_pd)}')


    # Create copies of the original training, testing, and validation datasets for each label
    original_angle_train_pd = angle_train_pd.copy()
    original_np_train_pd = np_train_pd.copy()
    original_pp1_train_pd = pp1_train_pd.copy()
    original_ep1_train_pd = ep1_train_pd.copy()
    
    original_angle_test_pd = angle_test_pd.copy()
    original_np_test_pd = np_test_pd.copy()
    original_pp1_test_pd = pp1_test_pd.copy()
    original_ep1_test_pd = ep1_test_pd.copy()
    
    original_angle_valid_pd = angle_valid_pd.copy()
    original_np_valid_pd = np_valid_pd.copy()
    original_pp1_valid_pd = pp1_valid_pd.copy()
    original_ep1_valid_pd = ep1_valid_pd.copy()
    
    # Lengths of the original train, test, and valid datasets    
    original_train_len = len(angle_train_pd)
    original_test_len = len(angle_test_pd)
    original_valid_len = len(angle_valid_pd)
    
    # Total length of the original training dataset
    total_train_len = len(original_angle_train_pd)
    
    # Loop to create 5-fold cross-validation datasets
    for i in range(5):
        # For subsequent iterations, create new validation and test datasets
        if i > 0:
            # Calculate start and end indices for slicing the training data
            start_idx = (i - 1) * (original_test_len + original_valid_len)
            end_idx = start_idx + original_test_len + original_valid_len
            
            # Ensure the end index does not exceed the length of the training data
            if end_idx > original_train_len:
                end_idx = original_train_len
            
            # Generate new validation and test datasets for 'angle'
            new_valid_test = original_angle_train_pd[start_idx:end_idx].reset_index(drop=True)
            angle_valid_pd = new_valid_test[:original_valid_len].reset_index(drop=True)
            angle_test_pd = new_valid_test[original_valid_len:].reset_index(drop=True)
            
            # Generate a new training dataset for 'angle'
            angle_train_pd = pd.concat([
                original_angle_train_pd[:start_idx], 
                original_angle_train_pd[end_idx:], 
                original_angle_valid_pd, 
                original_angle_test_pd
            ]).reset_index(drop=True)

            # Repeat the process for 'NP', 'PP1', and 'EP1' labels
            new_valid_test = original_np_train_pd[start_idx:end_idx].reset_index(drop=True)
            np_valid_pd = new_valid_test[:original_valid_len].reset_index(drop=True)
            np_test_pd = new_valid_test[original_valid_len:].reset_index(drop=True)
            
            np_train_pd = pd.concat([
                original_np_train_pd[:start_idx], 
                original_np_train_pd[end_idx:], 
                original_np_valid_pd, 
                original_np_test_pd
            ]).reset_index(drop=True)
            
            
            new_valid_test = original_pp1_train_pd[start_idx:end_idx].reset_index(drop=True)
            pp1_valid_pd = new_valid_test[:original_valid_len].reset_index(drop=True)
            pp1_test_pd = new_valid_test[original_valid_len:].reset_index(drop=True)
            
            pp1_train_pd = pd.concat([
                original_pp1_train_pd[:start_idx], 
                original_pp1_train_pd[end_idx:], 
                original_pp1_valid_pd, 
                original_pp1_test_pd
            ]).reset_index(drop=True)
            
            
            new_valid_test = original_ep1_train_pd[start_idx:end_idx].reset_index(drop=True)
            ep1_valid_pd = new_valid_test[:original_valid_len].reset_index(drop=True)
            ep1_test_pd = new_valid_test[original_valid_len:].reset_index(drop=True)
            
            ep1_train_pd = pd.concat([
                original_ep1_train_pd[:start_idx], 
                original_ep1_train_pd[end_idx:], 
                original_ep1_valid_pd, 
                original_ep1_test_pd
            ]).reset_index(drop=True)

        
        # Create directories for saving CSV files for the current fold
        uniform_data_path = create_dir(os.path.join(os.getcwd(), './datasets/cross_validation_uniform_data_'+str(i+1)))
        
        # Directories for each label within the fold
        uniform_data_angle_path = create_dir(os.path.join(uniform_data_path, 'angle'))
        uniform_data_pp1_path = create_dir(os.path.join(uniform_data_path, 'pp1'))
        uniform_data_ep1_path = create_dir(os.path.join(uniform_data_path, 'ep1'))
        uniform_data_np_path = create_dir(os.path.join(uniform_data_path, 'np'))
        
        # Paths for CSV files for each label
        uniform_angle_train_path = os.path.join(uniform_data_angle_path, 'train.csv')
        uniform_angle_valid_path = os.path.join(uniform_data_angle_path, 'valid.csv')
        uniform_angle_test_path = os.path.join(uniform_data_angle_path, 'test.csv')

        uniform_pp1_train_path = os.path.join(uniform_data_pp1_path, 'train.csv')
        uniform_pp1_valid_path = os.path.join(uniform_data_pp1_path, 'valid.csv')
        uniform_pp1_test_path = os.path.join(uniform_data_pp1_path, 'test.csv')

        uniform_np_train_path = os.path.join(uniform_data_np_path, 'train.csv')
        uniform_np_valid_path = os.path.join(uniform_data_np_path, 'valid.csv')
        uniform_np_test_path = os.path.join(uniform_data_np_path, 'test.csv')

        uniform_ep1_train_path = os.path.join(uniform_data_ep1_path, 'train.csv')
        uniform_ep1_valid_path = os.path.join(uniform_data_ep1_path, 'valid.csv')
        uniform_ep1_test_path = os.path.join(uniform_data_ep1_path, 'test.csv')

        # Create CSV files for each dataset and label
        create_csv(angle_train_pd, dataset, uniform_angle_train_path, label = 'angle')
        create_csv(angle_valid_pd, dataset, uniform_angle_valid_path, label = 'angle')
        create_csv(angle_test_pd, dataset, uniform_angle_test_path, label = 'angle')

        create_csv(pp1_train_pd, dataset, uniform_pp1_train_path, label = 'PP1')
        create_csv(pp1_valid_pd, dataset, uniform_pp1_valid_path, label = 'PP1')
        create_csv(pp1_test_pd, dataset, uniform_pp1_test_path, label = 'PP1')

        create_csv(np_train_pd, dataset, uniform_np_train_path, label = 'NP')
        create_csv(np_valid_pd, dataset, uniform_np_valid_path, label = 'NP')
        create_csv(np_test_pd, dataset, uniform_np_test_path, label = 'NP')

        create_csv(ep1_train_pd, dataset, uniform_ep1_train_path, label = 'EP1')
        create_csv(ep1_valid_pd, dataset, uniform_ep1_valid_path, label = 'EP1')
        create_csv(ep1_test_pd, dataset, uniform_ep1_test_path, label = 'EP1')