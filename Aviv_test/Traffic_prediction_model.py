import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import subprocess
import wandb
import math
from Aviv_test.Traffic_prediction_utils import generate_synthetic_data, generate_arrival_matrix_synthetic, preprocess_synthetic_data


class GRUNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1, dropout_prob=0.5):
        super(GRUNetwork, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob)
        self.bn_gru = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        out, _ = self.gru(x, h0)
        out = out[:, -1, :]
        # out = self.bn_gru(out[:, -1, :])
        # out = self.dropout(out)
        out = self.fc(out)
        return out


class LSTMNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1, dropout_prob=0.5):
        super(LSTMNetwork, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob)

        # Batch normalization and dropout
        self.bn_lstm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn_fc1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Take the output of the last time step

        out = self.bn_lstm(out)
        out = self.dropout(out)

        out = self.fc1(out)
        out = self.bn_fc1(out)
        out = self.dropout(out)

        out = self.fc2(out)
        return out


class TransformerNetwork(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, dropout_prob=0.5, output_dim=1):
        super(TransformerNetwork, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout_prob)

        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout_prob)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        self.fc1 = nn.Linear(d_model, d_model)
        self.bn_fc1 = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(d_model, output_dim)

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        output = self.transformer_encoder(src)
        output = output[:, -1, :]  # Take the output of the last time step

        output = self.fc1(output)
        output = self.bn_fc1(output)
        output = self.dropout(output)

        output = self.fc2(output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def train(X_train, y_train, X_val, y_val, model, model_name, num_epochs, batch_size, optimizer, criterion, lr, weight_decay,
          device, print_every=10, save_every=10, load_model=False, WANDB_TRACKING=True
          ):
    # ------------ WANDB Tracking----------------------#
    if WANDB_TRACKING:
        wandb.login(key="3ec39d34b7882297a057fdc2126cd037352175a4")
        wandb.init(project="Network_Prediction_Model",
                   config={
                       "epochs": num_epochs,
                       "batch_size": batch_size,
                       "lr": lr,
                    })

    # --------------- Defining Directories ----------------
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the save directory relative to the base directory
    save_dir = os.path.join(base_dir, 'Traffic_prediction_models')
    os.makedirs(save_dir, exist_ok=True)

    # ---------------------------------------------#

    if load_model:
        model_path = os.path.join(save_dir, f'{model_name}\\model_best_model.pk')
        model.load_state_dict(torch.load(model_path, map_location=device))

    model = model.to(device)

    optimizer = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best_model = model
    best_val_loss = np.inf
    best_epoch = 0
    # Initializing arrays for plotting
    train_loss = []
    validation_loss = []

    # start training loop
    for epoch in range(num_epochs):

        model.train()
        running_loss = 0
        permutation = torch.randperm(X_train.size(0))

        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            X_batch, y_batch = X_train[indices], y_train[indices]

            # len X_batch doesn't surely divide by batch size, we skip last minibatch
            if np.shape(X_batch)[0] != batch_size:
                continue

            optimizer.zero_grad()

            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch.unsqueeze(-1))
            # loss = criterion(output, y_batch.unsqueeze(-1))
            loss = torch.sqrt_(criterion(output, y_batch.unsqueeze(-1)))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            running_loss += loss.item()

        running_loss /= (len(X_train) // batch_size)  # Average loss
        train_loss.append(running_loss)

        # Evaluating the model
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for i in range(0, X_val.size(0), batch_size):
                X_batch, y_batch = X_val[i:i + batch_size], y_val[i:i + batch_size]
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                # len X_batch doesn't surely divide by batch size, we skip last minibatch
                if np.shape(X_batch)[0] != batch_size:
                    continue

                val_output = model(X_batch.unsqueeze(-1))
                running_loss += torch.sqrt_(criterion(val_output, y_batch.unsqueeze(-1))).item()
                # running_loss += criterion(val_output, y_batch.unsqueeze(-1)).item()
        running_loss /= (len(X_val) // batch_size)
        validation_loss.append(running_loss)

        scheduler.step(running_loss)  # Step the scheduler with the validation loss

        if WANDB_TRACKING:
            wandb.log({"Train Loss": train_loss[-1], "Val Loss": validation_loss[-1],
                       "Learning Rate": optimizer.param_groups[0]['lr']})

        if (epoch + 1) % print_every == 0:
            print(
                f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss[-1]:.4f} , Validation Loss: {validation_loss[-1]:.4f}')

        if (epoch + 1) % save_every == 0:
            # Construct the checkpoint path, save weights every few epochs
            checkpoint_path = os.path.join(save_dir, f'{model_name}')
            checkpoint_path = os.path.join(checkpoint_path, f'model_{epoch+1}.pk')
            torch.save(best_model.state_dict(), checkpoint_path)

        # Check if model improved
        if validation_loss[-1] < best_val_loss:
            best_val_loss = validation_loss[-1]
            best_model = model
            best_epoch = epoch

    # save best model
    checkpoint_path = os.path.join(save_dir, f'{model_name}')
    checkpoint_path = os.path.join(checkpoint_path, f'model_best_model.pk')
    torch.save(best_model.state_dict(), checkpoint_path)

    print(f'Finished Training, best model was epoch: {best_epoch}')

    if WANDB_TRACKING:
       wandb.finish()

    return np.array(train_loss), np.array(validation_loss), best_model


def test_model(model, X_test, y_test, criterion, device='cpu'):
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode

    test_loss = 0.0
    with torch.no_grad():  # Disable gradient calculation for efficiency
        for i in range(X_test.size(0)):
            data, targets = X_test[i].to(device), y_test[i].to(device)
            outputs = model(data.unsqueeze(0).unsqueeze(-1))
            loss = torch.sqrt_(criterion(outputs, targets.unsqueeze(0).unsqueeze(-1)))
            test_loss += loss.item()

    test_loss = test_loss / X_test.size(0)  # Average loss

    print(f'Test Loss: {test_loss:.4f}')

    return test_loss


def MA_test(model, X_test, y_test, device='cpu'):
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode

    absolute_errors = []
    with torch.no_grad():  # Disable gradient calculation for efficiency
        for i in range(X_test.size(0)):
            data, targets = X_test[i].to(device), y_test[i].to(device)
            outputs = model(data.unsqueeze(0).unsqueeze(-1))
            absolute_error = torch.abs((targets.unsqueeze(0).unsqueeze(-1) - outputs) / targets.unsqueeze(0).unsqueeze(-1))
            absolute_errors.append(absolute_error.item())

    mean_absolute_error = np.mean(absolute_errors)
    ma = (1 - mean_absolute_error) * 100

    print(f'Mean Accuracy (MA): {ma:.4f}%')
    return ma


def plot_curves(train_losses, val_losses, model='LSTM'):
    plt.figure()

    plt.plot(np.arange(len(train_losses)), train_losses, label='Train')
    plt.plot(np.arange(len(val_losses)), val_losses, label='Validation')

    plt.title(f"RMSE Loss Curves {model} Model")
    plt.grid(True)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Ensure the directory exists
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Traffic_prediction_loss_curves')
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f'training_{model}.png')
    plt.savefig(save_path)
    plt.close()


def NTP(model_type='LSTM',training=False, test=False, generate_matrix=True, num_flows=10, num_time_steps=60):

    csv_file = r'C:\Users\beaviv\Datasets\packets.csv'
    # specify the number of colomns we want to take from the csv file
    data_points_number = 1e4
    # df = read_and_clean_csv(file_path=csv_file, expected_fields=3, data_points_number=data_points_number)
    # df, scaler = preprocess_data(df=df)
    # X_train, X_val, X_test, y_train, y_val, y_test = prepare_train_val_test_data(df=df, seq_length=20)

    # Generate synthetic data
    synthetic_data = generate_synthetic_data()

    # Preprocess synthetic data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_synthetic_data(synthetic_data, seq_length=20)

    model_parameters = {'input_dim': 1,
                        'hidden_dim': 120,
                        'output_dim': 1,
                        'num_layers': 4,
                        'dropout_prob': 0.6}

    if model_type == 'LSTM':
        model = LSTMNetwork(**model_parameters)
    elif model_type == 'GRU':
        model = GRUNetwork(**model_parameters)
    elif model_type == 'Transformer':
        model_parameters = {'input_dim': 1, 'd_model': 64, 'nhead': 4, 'num_layers': 2, 'dim_feedforward': 256,
                            'dropout_prob': 0.5, 'output_dim': 1}
        model = TransformerNetwork(**model_parameters)
    else:
        raise ValueError("Unsupported model type. Choose 'LSTM' or 'GRU'.")

    """
    Test and generate prediction
    """
    if test or generate_matrix:
        model_path = r'C:\Users\beaviv\DIAMOND\Aviv_test'
        model_path = os.path.join(model_path, f'Traffic_prediction_models\\{model_type}\\model_best_model.pk')
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))

        if test:
            test_loss = test_model(model, X_test, y_test, nn.MSELoss(), device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        if generate_matrix:
            # Generate arrival matrix using the test set data
            # test_indices = np.arange(len(df) - len(X_test), len(df))
            # test_df = df.iloc[test_indices]

            # arrival_matrix = generate_arrival_matrix(model, scaler, test_df, num_flows=num_flows, num_time_steps=num_time_steps
            #                                          , device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

            # Generate arrival matrix using the test set data
            # test_indices = np.arange(len(X_test))
            # test_data = synthetic_data[:, test_indices].T  # Transpose to match the original data shape

            arrival_matrix = generate_arrival_matrix_synthetic(model, X_test,
                                                               num_flows=num_flows,
                                                               num_time_steps=num_time_steps,
                                                               seq_length=20,
                                                               device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                                                               )

            return arrival_matrix

    """
    Train
    """
    if training:
        # Hyperparameters
        train_hyperparameters = {'model': model,
                                 'model_name': model_type,
                                 'X_train': X_train,
                                 'y_train': y_train,
                                 'X_val': X_val,
                                 'y_val': y_val,
                                 'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                                 'optimizer': torch.optim.Adam,
                                 'criterion': nn.MSELoss(),
                                 'num_epochs': 300,
                                 'batch_size': 32,
                                 'lr': 0.0001,
                                 'print_every': 10,
                                 'save_every': 20,
                                 'weight_decay': 1e-3,
                                 'load_model': False,
                                 'WANDB_TRACKING': True
                                 }

        train_loss, validation_loss, best_model = train(**train_hyperparameters)
        plot_curves(train_losses=train_loss, val_losses=validation_loss, model=model_type)


if __name__ == "__main__":
    NTP(model_type='LSTM', training=True, test=False, generate_matrix=False)
