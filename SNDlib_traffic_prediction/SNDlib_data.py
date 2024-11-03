import os
import pandas as pd
import numpy as np
from lxml import etree
import torch
from matplotlib import pyplot as plt


def process_traffic_matrices(directory_path):
    # Create a set to store all the unique links
    unique_links = set()

    # Get a list of all XML files in the directory
    xml_files = sorted([f for f in os.listdir(directory_path) if f.endswith('.xml')])

    # Initialize the dataframe with zeros
    num_time_steps = len(xml_files)

    # If no files are found, return an empty dataframe
    if num_time_steps == 0:
        return pd.DataFrame()

    # Process the first file to get the links and initialize the dataframe
    first_file = os.path.join(directory_path, xml_files[0])
    tree = etree.parse(first_file)
    demands = tree.xpath('//ns:demand', namespaces={'ns': 'http://sndlib.zib.de/network'})
    for demand in demands:
        source = demand.xpath('ns:source/text()', namespaces={'ns': 'http://sndlib.zib.de/network'})[0]
        target = demand.xpath('ns:target/text()', namespaces={'ns': 'http://sndlib.zib.de/network'})[0]
        link = f"{source}_{target}"
        unique_links.add(link)

    # Create a sorted list of links
    links = sorted(unique_links)

    # Initialize the dataframe with the identified links as columns
    num_links = len(links)
    df = pd.DataFrame(np.zeros((num_time_steps, num_links)), columns=links)

    # Process each XML file and update the dataframe
    for time_step, filename in enumerate(xml_files):
        file_path = os.path.join(directory_path, filename)
        tree = etree.parse(file_path)
        demands = tree.xpath('//ns:demand', namespaces={'ns': 'http://sndlib.zib.de/network'})
        for demand in demands:
            source = demand.xpath('ns:source/text()', namespaces={'ns': 'http://sndlib.zib.de/network'})[0]
            target = demand.xpath('ns:target/text()', namespaces={'ns': 'http://sndlib.zib.de/network'})[0]
            link = f"{source}_{target}"
            value = float(
                demand.xpath('ns:demandValue/text()', namespaces={'ns': 'http://sndlib.zib.de/network'})[0].strip())
            df.at[time_step, link] = value

    return df


# For Geant dataset
def process_traffic_matrices_2(directory_path):
    # Create a set to store all the unique links
    unique_links = set()

    # Get a list of all XML files in the directory
    xml_files = sorted([f for f in os.listdir(directory_path) if f.endswith('.xml')])

    # Initialize a list to store data for the dataframe
    data = []

    # Process the first file to get the links and initialize the dataframe
    for filename in xml_files:
        file_path = os.path.join(directory_path, filename)
        tree = etree.parse(file_path)
        demands = tree.xpath('//ns:demand', namespaces={'ns': 'http://sndlib.zib.de/network'})

        # Skip if no demands are found
        if not demands:
            print(f"No demand data found in {filename}. Skipping this file.")
            continue

        # Extract all unique links (source_target combinations)
        row_data = {}
        for demand in demands:
            source = demand.xpath('ns:source/text()', namespaces={'ns': 'http://sndlib.zib.de/network'})[0]
            target = demand.xpath('ns:target/text()', namespaces={'ns': 'http://sndlib.zib.de/network'})[0]
            link = f"{source}_{target}"
            unique_links.add(link)

            value = float(
                demand.xpath('ns:demandValue/text()', namespaces={'ns': 'http://sndlib.zib.de/network'})[0].strip())
            row_data[link] = value

        # Add row data (if demands exist) to the list
        data.append(row_data)

    # Convert the list of row_data dictionaries into a dataframe
    df = pd.DataFrame(data)

    # Ensure all unique links are present as columns (even if missing in some rows)
    df = df.reindex(columns=sorted(unique_links), fill_value=np.nan)

    # Replace None (NaN) with 0 in the dataframe
    df.fillna(0, inplace=True)

    return df


def artificial_data_csv(save_directory, csv_filename, artificial_csv_filename, num_time_steps):

    # Step 1: Load the original DataFrame to understand its structure

    df_csv = load_dataframe_from_csv(save_directory, csv_filename)

    # Step 2: Generate artificial data
    num_links = df_csv.shape[1]  # Number of links (columns) should remain the same

    # Generate artificial data using random numbers
    # You can adjust the distribution and mean/std based on your requirements
    artificial_data = np.random.normal(loc=20, scale=10, size=(num_time_steps, num_links))

    # Create a new DataFrame with the same columns as the original
    df_artificial = pd.DataFrame(artificial_data, columns=df_csv.columns)

    # Step 3: Save the new DataFrame as a CSV file
    save_dataframe_as_csv(df=df_artificial, directory=save_directory, filename=artificial_csv_filename)


def save_dataframe_as_csv(df, directory, filename):
    os.makedirs(directory, exist_ok=True)  # Ensure the directory exists
    file_path = os.path.join(directory, filename)
    df.to_csv(file_path, index=False)


def load_dataframe_from_csv(directory, filename):
    file_path = os.path.join(directory, filename)
    return pd.read_csv(file_path)


def split_dataframe(df, train_ratio=0.6, val_ratio=0.1, test_ratio=0.3):
    # Ensure the ratios sum up to 1
    # assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum up to 1"

    # Calculate the number of rows for each set
    total_rows = len(df)
    train_end = int(total_rows * train_ratio)
    val_end = train_end + int(total_rows * val_ratio)

    # Split the dataframe
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    return train_df, val_df, test_df


def create_sequences(data, input_length=72, output_length=12):
    X, y = [], []
    for i in range(len(data) - input_length - output_length + 1):
        X.append(data[i:i+input_length].values)
        y.append(data[i+input_length:i+input_length+output_length].values)
    return np.array(X), np.array(y)


def create_sequences_single_pass(data, input_length=72, output_length=12):
    X, y = [], []
    i = 0
    while i + input_length + output_length <= len(data):
        # Select X and y based on input and output lengths
        X.append(data[i:i + input_length].values)
        y.append(data[i + input_length:i + input_length + output_length].values)

        # Move to the next block of data (next 84 rows)
        i += input_length + output_length

    return np.array(X), np.array(y)


def prepare_datasets(df, train_ratio=0.6, val_ratio=0.1, test_ratio=0.3, input_length=72, output_length=12):
    # Split the dataframe
    train_df, val_df, test_df = split_dataframe(df, train_ratio, val_ratio, test_ratio)

    # Create sequences
    X_train, y_train = create_sequences(train_df, input_length, output_length)
    X_val, y_val = create_sequences(val_df, input_length, output_length)
    X_test, y_test = create_sequences(test_df, input_length, output_length)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    return X_train, y_train, X_val, y_val, X_test, y_test


def normalize_data(X_train, X_val, X_test, y_train, y_val, y_test, type='min_max'):

    if type == 'gaussian':
        # Compute mean and std for both inputs and targets using the training set
        train_mean = X_train.mean(dim=(0, 1), keepdim=True)
        train_std = X_train.std(dim=(0, 1), keepdim=True)

        # Normalize inputs
        X_train_normalized = (X_train - train_mean) / train_std
        X_val_normalized = (X_val - train_mean) / train_std
        X_test_normalized = (X_test - train_mean) / train_std

        # Normalize targets
        y_train_normalized = (y_train - train_mean) / train_std
        y_val_normalized = (y_val - train_mean) / train_std
        y_test_normalized = (y_test - train_mean) / train_std

    if type == 'min_max':
        # Compute min and max values across batches and time steps for each feature (using the training data)
        train_min = X_train.amin(dim=(0, 1), keepdim=True)
        train_max = X_train.amax(dim=(0, 1), keepdim=True)

        # Apply min-max normalization to inputs
        X_train_normalized = (X_train - train_min) / (train_max - train_min)
        X_val_normalized = (X_val - train_min) / (train_max - train_min)
        X_test_normalized = (X_test - train_min) / (train_max - train_min)

        # Apply min-max normalization to targets
        y_train_normalized = (y_train - train_min) / (train_max - train_min)
        y_val_normalized = (y_val - train_min) / (train_max - train_min)
        y_test_normalized = (y_test - train_min) / (train_max - train_min)

        # Handle potential division by zero where max equals min for both inputs and targets
        X_train_normalized = torch.nan_to_num(X_train_normalized, nan=0.0)
        X_val_normalized = torch.nan_to_num(X_val_normalized, nan=0.0)
        X_test_normalized = torch.nan_to_num(X_test_normalized, nan=0.0)

        y_train_normalized = torch.nan_to_num(y_train_normalized, nan=0.0)
        y_val_normalized = torch.nan_to_num(y_val_normalized, nan=0.0)
        y_test_normalized = torch.nan_to_num(y_test_normalized, nan=0.0)

    return X_train_normalized, X_val_normalized, X_test_normalized, y_train_normalized, y_val_normalized, y_test_normalized


def denormalize_predictions(predictions, train_min_or_mean, train_max_or_std, type='min_max'):
    if type == 'min_max':
        predictions_denormalized = predictions * (train_max_or_std.numpy() - train_min_or_mean.numpy()) + train_min_or_mean.numpy()
    elif type == 'gaussian':
        predictions_denormalized = predictions * train_max_or_std.numpy() + train_min_or_mean.numpy()
    return predictions_denormalized[0]


def view_data(df_csv):
    # Calculate the average traffic volume per time step
    average_traffic_volume = df_csv.mean(axis=1)  # Mean across all columns (links) for each row (time step)

    plt.figure(figsize=(12, 6))
    plt.plot(average_traffic_volume)
    plt.title('Average Traffic Volume per Time Step')
    plt.xlabel('Time Step')
    plt.ylabel('Average Traffic Volume')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # save_directory = r'C:\Users\beaviv\Datasets\SNDlib'
    save_directory = r'C:\Users\aviv9\Datasets\SNDlib'

    # Abilene
    # abilene_directory_path = r'C:\Users\aviv9\Datasets\SNDlib\directed-abilene-zhang-5min-over-6months-ALL'
    # abilene_csv_filename = 'abilene_traffic_matrices.csv'
    # abilene_df = process_traffic_matrices(directory_path=abilene_directory_path)
    # save_dataframe_as_csv(df=abilene_df, directory=save_directory, filename=abilene_csv_filename)

    # Geant
    geant_directory_path = r'C:\Users\aviv9\Datasets\SNDlib\directed-geant-uhlig-15min-over-4months-ALL'
    geant_csv_filename = 'geant_traffic_matrices.csv'
    geant_df = process_traffic_matrices_2(directory_path=geant_directory_path)
    save_dataframe_as_csv(df=geant_df, directory=save_directory, filename=geant_csv_filename)

    # artificial_csv_filename = 'artificial_traffic_matrices.csv'

    # View Data
    df_csv = load_dataframe_from_csv(save_directory, geant_csv_filename)
    # Remove the time steps with the spike
    geant_df_cleaned = geant_df.drop(df_csv.index[5136:5148])

    view_data(geant_df_cleaned)

    # artificial_data_csv(save_directory=save_directory, csv_filename=csv_filename,
    #                     artificial_csv_filename=artificial_csv_filename,
    #                     num_time_steps=1000)

    # View Artificial Data
    # artificial_df_csv = load_dataframe_from_csv(save_directory, artificial_csv_filename)
    # view_data(artificial_df_csv)