import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os


def preprocess_data(df):
    # Check if the necessary columns are present
    required_columns = ['ip.src', 'ip.dst', 'frame.len']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise KeyError(f"Missing columns in the CSV file: {', '.join(missing_columns)}")

    # Add synthetic time column
    df['time'] = pd.date_range(start='2021-01-01 14:00:00', periods=len(df), freq='S')

    # Convert IPs to categorical codes
    df.loc[:, 'ip.src'] = df['ip.src'].astype('category').cat.codes
    df.loc[:, 'ip.dst'] = df['ip.dst'].astype('category').cat.codes

    # Convert 'frame.len' to integers
    df.loc[:, 'frame.len'] = df['frame.len'].astype(int)

    # Inspect the number of unique values
    num_unique_ip_src = df['ip.src'].nunique()
    num_unique_ip_dst = df['ip.dst'].nunique()

    print(f"Number of unique ip.src: {num_unique_ip_src}")
    print(f"Number of unique ip.dst: {num_unique_ip_dst}")

    # Limit the number of unique values to avoid overflow
    if num_unique_ip_src * num_unique_ip_dst > 10000:
        print("Reducing the number of unique source-destination pairs to avoid overflow...")
        top_srcs = df['ip.src'].value_counts().nlargest(100).index
        top_dsts = df['ip.dst'].value_counts().nlargest(100).index
        df = df[df['ip.src'].isin(top_srcs) & df['ip.dst'].isin(top_dsts)]

    # Ensure the 'time' column is set as DatetimeIndex
    df.loc[:, 'time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)

    # Normalize the packet sizes
    scaler = MinMaxScaler()
    df['frame.len'] = scaler.fit_transform(df[['frame.len']])
    # TODO:  don't want to normalize packet size
    return df, scaler  # return df


def read_and_clean_csv(file_path, expected_fields=3, data_points_number=2e5):
    valid_lines = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        header = file.readline().strip().split(',')
        valid_lines.append(header)
        for line in file:
            # Get specified number of data points
            if len(valid_lines) >= data_points_number + 1:
                break
            fields = line.strip().split(',')
            if len(fields) == expected_fields and all(fields):
                valid_lines.append(fields)

    # Convert to DataFrame
    df = pd.DataFrame(valid_lines[1:], columns=valid_lines[0])  # Skip the header row
    return df


def preprocess_data(df):
    # Check if the necessary columns are present
    required_columns = ['ip.src', 'ip.dst', 'frame.len']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise KeyError(f"Missing columns in the CSV file: {', '.join(missing_columns)}")

    # Add synthetic time column
    df['time'] = pd.date_range(start='2021-01-01 14:00:00', periods=len(df), freq='S')

    # Convert IPs to categorical codes
    df.loc[:, 'ip.src'] = df['ip.src'].astype('category').cat.codes
    df.loc[:, 'ip.dst'] = df['ip.dst'].astype('category').cat.codes

    # Convert 'frame.len' to integers
    df.loc[:, 'frame.len'] = df['frame.len'].astype(int)

    # Inspect the number of unique values
    num_unique_ip_src = df['ip.src'].nunique()
    num_unique_ip_dst = df['ip.dst'].nunique()

    print(f"Number of unique ip.src: {num_unique_ip_src}")
    print(f"Number of unique ip.dst: {num_unique_ip_dst}")

    # Limit the number of unique values to avoid overflow
    if num_unique_ip_src * num_unique_ip_dst > 10000:
        print("Reducing the number of unique source-destination pairs to avoid overflow...")
        top_srcs = df['ip.src'].value_counts().nlargest(100).index
        top_dsts = df['ip.dst'].value_counts().nlargest(100).index
        df = df[df['ip.src'].isin(top_srcs) & df['ip.dst'].isin(top_dsts)]

    # Ensure the 'time' column is set as DatetimeIndex
    df.loc[:, 'time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)

    # Normalize the packet sizes
    scaler = MinMaxScaler()
    df['frame.len'] = scaler.fit_transform(df[['frame.len']])
    # TODO:  don't want to normalize packet size
    return df, scaler  # return df


def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def prepare_train_val_test_data(df, seq_length=10):
    flows = df.groupby(['ip.src', 'ip.dst'])
    X, y = [], []

    for _, group in flows:
        group_values = group['frame.len'].values
        X_seq, y_seq = create_sequences(data=group_values, seq_length=seq_length)

        # Filter out empty arrays
        if X_seq.size > 0 and y_seq.size > 0:
            X.append(X_seq)
            y.append(y_seq)

    if len(X) == 0 or len(y) == 0:
        raise ValueError("No valid sequences found. Please check your data and sequence length.")

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, shuffle=True)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=True)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    return X_train, X_val, X_test, y_train, y_val, y_test


def generate_synthetic_data(num_samples=5000, num_flows=10, burst_size=200, burst_frequency=100,
                                   noise_level=0.1):
    time_series_data = []
    for _ in range(num_flows):
        # Create a baseline sawtooth wave
        time = np.arange(num_samples)
        period = 50  # Period of the sawtooth wave
        sawtooth_wave = (time % period) / period * 100  # Normalize to a range, e.g., 0-100

        # Add baseline noise
        noise = np.random.normal(loc=0, scale=noise_level * 100, size=num_samples)  # Adjust noise scale accordingly
        baseline = sawtooth_wave + noise

        # Add bursts of high activity
        burst_indices = np.random.choice(np.arange(num_samples), size=num_samples // burst_frequency, replace=False)
        for index in burst_indices:
            start = max(0, index - burst_size // 2)
            end = min(num_samples, index + burst_size // 2)
            baseline[start:end] += np.random.uniform(low=100, high=300,
                                                     size=end - start)  # Bursts with packet sizes between 200 and 500

        # Ensure no negative values
        baseline = np.clip(baseline, 0, None)
        time_series_data.append(baseline)

        if len(time_series_data) == 1:
            plt.plot(time, time_series_data[0])
            plt.grid(True)
            plt.xlabel("Time Step")
            plt.ylabel('Packet Volume')
            plt.title('Synthetic Data generation')
            plt.show()

    return np.array(time_series_data)


def preprocess_synthetic_data(data, seq_length=20):
    X, y = [], []
    for series in data:
        X_seq, y_seq = create_sequences(series, seq_length)
        X.append(X_seq)
        y.append(y_seq)

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, shuffle=True)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=True)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    return X_train, X_val, X_test, y_train, y_val, y_test


def predict_next_packet(model, input_seq, device='cpu'):
    model.eval()
    model=model.to(device=device)
    with torch.no_grad():
        input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        prediction = model(input_tensor)
    return prediction.item()


def generate_arrival_matrix(model, scaler, df, num_flows, num_time_steps, device='cpu'):
    arrival_matrix = np.zeros((num_time_steps, num_flows))

    # Identify the top N flows
    flow_counts = df.groupby(['ip.src', 'ip.dst']).size().reset_index(name='counts')
    top_flows = flow_counts.nlargest(num_flows, 'counts')[['ip.src', 'ip.dst']]

    for i, (src, dst) in enumerate(zip(top_flows['ip.src'], top_flows['ip.dst'])):
        flow_data = df[(df['ip.src'] == src) & (df['ip.dst'] == dst)]['frame.len'].values
        for t in range(num_time_steps):
            if t < len(flow_data):
                input_seq = flow_data[max(0, t - 10):t]
            else:
                input_seq = arrival_matrix[t - 10:t, i]

            if len(input_seq) < 10:
                input_seq = np.pad(input_seq, (10 - len(input_seq), 0), 'constant')

            prediction = predict_next_packet(model, input_seq, device)
            arrival_matrix[t, i] = scaler.inverse_transform([[prediction]])[0][0]

    return arrival_matrix.astype('int64')


def generate_arrival_matrix_synthetic(model, X_test, num_flows, num_time_steps, seq_length, device='cpu'):
    arrival_matrix = np.zeros((num_time_steps, num_flows))

    for i in range(num_flows):
        flow_data = X_test[:, i].flatten()  # Flatten to get the full time series for the flow
        for t in range(num_time_steps):
            if t < seq_length:
                input_seq = flow_data[max(0, t - seq_length):t]
            else:
                input_seq = arrival_matrix[t - seq_length:t, i]

            if len(input_seq) < seq_length:
                input_seq = np.pad(input_seq, (seq_length - len(input_seq), 0), 'constant')

            prediction = predict_next_packet(model, input_seq, device)
            arrival_matrix[t, i] = prediction

    return arrival_matrix.astype('float32')  # Use float32 to avoid integer truncation issues

































