import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data, Batch
import torch.optim as optim
from SNDlib_data import load_dataframe_from_csv, prepare_datasets
from environment.utils import create_abilene_graph, create_snd_geant_graph
import networkx as nx
import numpy as np
from torch_geometric.utils import from_networkx
from scipy.spatial.distance import pdist, squareform
from SNDlib_data import normalize_data


class GRUPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(GRUPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size, output_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        batch_size, time_steps, num_flows = x.size()

        # h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = out[:, -1, :]  # Get the output of the last time step
        out = self.batch_norm(out)
        out = self.fc(out)
        out = out.view(out.size(0), 12, num_flows)  # Reshape the output to (batch_size, 12, num_flows)
        return out


class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size, output_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        batch_size, time_steps, num_flows = x.size()

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Get the output of the last time step
        out = self.batch_norm(out)
        out = self.fc(out)
        out = out.view(out.size(0), 12, num_flows)  # Reshape the output to (batch_size, 12, num_flows)
        return out


class RNNPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNNPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size, output_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        batch_size, time_steps, num_flows = x.size()

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]  # Get the output of the last time step
        out = self.batch_norm(out)
        out = self.fc(out)
        out = out.view(out.size(0), 12, num_flows)  # Reshape the output to (batch_size, 12, num_flows)
        return out


class MLPPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(MLPPredictor, self).__init__()
        self.fc1 = nn.Linear(72 * input_size, hidden_size)  # 72*len(df.columns) the input size matrix
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.bath_norm1 = nn.BatchNorm1d(hidden_size)
        self.bath_norm2 = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        batch_size, time_steps, num_flows = x.size()

        x = x.view(x.size(0), -1)  # Flatten the input for the MLP
        out = torch.relu(self.bath_norm1(self.fc1(x)))
        out = self.dropout1(out)
        out = torch.relu(self.bath_norm2(self.fc2(out)))
        out = self.dropout2(out)
        out = self.fc3(out)
        out = out.view(out.size(0), 12, num_flows)  # Reshape the output to (batch_size, 12, num_flows)
        return out


class GNNPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, edge_index, num_gcn_layers):
        super(GNNPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_gcn_layers = num_gcn_layers

        # Define multiple GCN layers using ModuleList
        self.gcn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()  # Define dropout layers

        # Add the first GCN layer
        self.gcn_layers.append(pyg_nn.GCNConv(input_size, hidden_size))
        self.batch_norms.append(nn.BatchNorm1d(hidden_size))
        self.dropouts.append(nn.Dropout(p=0.5))

        # Add additional GCN layers
        for _ in range(1, num_gcn_layers):
            self.gcn_layers.append(pyg_nn.GCNConv(hidden_size, hidden_size))
            self.batch_norms.append(nn.BatchNorm1d(hidden_size))
            self.dropouts.append(nn.Dropout(p=0.5))

        # GRU Layer for temporal processing
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)

        # LSTM Layer for temporal processing
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.5)

        # Fully Connected Layer to produce the final output
        self.fc = nn.Linear(hidden_size, output_size)

        # Store the graph structure (edge_index) for GCN
        self.edge_index = edge_index

    def forward(self, x):
        # x shape: (batch, time_steps, num_links)
        batch_size, time_steps, num_flows = x.size()

        # Reshape x to match the GCN input format
        x = x.view(batch_size * time_steps, num_flows)  # Flatten time steps into batch

        # Apply GCN layers sequentially with batch normalization and dropout
        for gcn, bn, dropout in zip(self.gcn_layers, self.batch_norms, self.dropouts):
            x = gcn(x, self.edge_index)
            x = bn(x)  # Apply batch normalization
            x = F.relu(x)  # Apply activation
            x = dropout(x)  # Apply dropout

        x = x.view(batch_size, time_steps, -1)  # Reshape back to (batch, time_steps, hidden_size)

        # Apply GRU layer
        # h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        # out, _ = self.gru(x, h0)

        # Initialize LSTM hidden and cell states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)  # Hidden state
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)  # Cell state

        # Apply LSTM layer
        out, _ = self.lstm(x, (h0, c0))

        # Apply the Fully Connected layer
        out = out[:, -1, :]  # Take the last time step output
        out = self.fc(out)
        out = out.view(batch_size, 12, num_flows)  # Reshape to (batch, 12, num_flows)
        return out


class GNNPredictorWithDistance(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, num_gcn_layers, edge_index):
        super(GNNPredictorWithDistance, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_gcn_layers = num_gcn_layers

        # Define multiple GCN layers using ModuleList
        self.gcn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()  # Define dropout layers

        # Add the first GCN layer
        self.gcn_layers.append(pyg_nn.GCNConv(input_size, hidden_size))
        self.batch_norms.append(nn.BatchNorm1d(hidden_size))
        self.dropouts.append(nn.Dropout(p=0.2))

        # Add additional GCN layers
        for _ in range(1, num_gcn_layers):
            self.gcn_layers.append(pyg_nn.GCNConv(hidden_size, hidden_size))
            self.batch_norms.append(nn.BatchNorm1d(hidden_size))
            self.dropouts.append(nn.Dropout(p=0.2))

        # LSTM Layer for temporal processing
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)

        # Fully Connected Layer to produce the final output
        self.fc = nn.Linear(hidden_size, output_size)

        # Store the graph structure (edge_index) for GCN
        self.edge_index = edge_index

    def forward(self, x, distance):
        # x shape: (batch, time_steps, num_links)
        # distance shape: (num_nodes, num_nodes) - reshape if necessary to match input
        batch_size, time_steps, num_links = x.size()

        # Reshape x to match the GCN input format and add distance information
        x = x.view(batch_size * time_steps, num_links)  # Flatten time steps into batch

        # Concatenate distance information to x
        # Assuming distance is added as a feature to each link (you may need to align shapes accordingly)

        # Expand the distance to match batch and time steps: (batch_size * time_steps, num_nodes, num_nodes)
        distance_expanded = distance.unsqueeze(0).repeat(batch_size * time_steps, 1, 1)
        distance_expanded = distance_expanded.view(batch_size * time_steps, -1)  # Adjust as needed
        x = torch.cat([x, distance_expanded], dim=1)  # Concatenate along the feature dimension

        # Apply GCN layers sequentially with batch normalization and dropout
        for gcn, bn, dropout in zip(self.gcn_layers, self.batch_norms, self.dropouts):
            x = gcn(x, self.edge_index)
            # x = bn(x)  # Apply batch normalization
            x = F.relu(x)  # Apply activation
            # x = dropout(x)  # Apply dropout

        x = x.view(batch_size, time_steps, -1)  # Reshape back to (batch, time_steps, hidden_size)

        # Initialize LSTM hidden and cell states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)  # Hidden state
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)  # Cell state

        # Apply LSTM layer
        out, _ = self.lstm(x, (h0, c0))

        # Apply the Fully Connected layer
        out = out[:, -1, :]  # Take the last time step output
        out = self.fc(out)
        out = out.view(batch_size, 12, num_links)  # Reshape to (batch, 12, 132)
        return out


class ACRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, attention_size, dropout_prob=0.5):
        super(ACRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention_size = attention_size

        # Correlational Modeling: CNN layers
        self.cnn_layers = nn.ModuleList()
        self.pooling_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # Define the convolutional and pooling layers
        for i in range(3):  # Three convolutional and pooling layers as specified
            in_channels = 1 if i == 0 else hidden_size
            self.cnn_layers.append(nn.Conv2d(in_channels, hidden_size, kernel_size=(3, 3), padding=1))
            self.batch_norms.append(nn.BatchNorm2d(hidden_size))
            self.pooling_layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
            self.dropouts.append(nn.Dropout(p=dropout_prob))

        # Determine the size after CNN and pooling layers dynamically
        self.flattened_size = self.calculate_flattened_size(input_size)

        # Inter-flow Attention Layer
        self.inter_flow_attention_w1 = nn.Linear(self.flattened_size, attention_size)
        self.inter_flow_attention_w2 = nn.Linear(self.flattened_size, attention_size)
        self.inter_flow_attention_out = nn.Linear(attention_size, self.flattened_size)

        # LSTM layer for Temporal Modeling
        self.lstm = nn.LSTM(self.flattened_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout_prob)

        # Intra-flow Attention Mechanism for LSTM outputs
        self.intra_flow_attention_v1 = nn.Linear(hidden_size, attention_size)
        self.intra_flow_attention_v2 = nn.Linear(hidden_size, attention_size)
        self.intra_flow_attention_out = nn.Linear(attention_size, hidden_size)

        # Fully Connected Layer to produce the final output
        self.fc = nn.Linear(hidden_size, output_size)

    def calculate_flattened_size(self, input_size):
        # Dummy input to calculate the output size after convolutions and pooling
        with torch.no_grad():
            x = torch.randn(1, 1, 72, input_size)  # Batch size 1, channels 1, temporal size 72, spatial size input_size
            for cnn, pool in zip(self.cnn_layers, self.pooling_layers):
                x = cnn(x)
                x = pool(x)
            return x.view(1, 72, -1).size(-1)  # Flattened size per time step

    def inter_flow_attention(self, x):
        # Inter-flow attention mechanism: applies attention on the CNN outputs
        w1 = self.inter_flow_attention_w1(x)  # Transform to attention size
        w2 = self.inter_flow_attention_w2(x)  # Transform to attention size
        s = torch.matmul(w1, w2.transpose(-1, -2)) / (self.attention_size ** 0.5)  # Compute attention scores, scaled
        s = F.softmax(s, dim=-1)  # Normalize scores
        x = torch.matmul(s, w1)  # Apply attention weights correctly
        return self.inter_flow_attention_out(x)  # Apply linear transformation to match the flattened size

    def intra_flow_attention(self, h):
        # Intra-flow attention mechanism: applies attention on LSTM hidden states
        v1 = self.intra_flow_attention_v1(h)  # Transform to attention size
        v2 = self.intra_flow_attention_v2(h)  # Transform to attention size
        e = torch.matmul(v1, v2.transpose(-1, -2)) / (self.attention_size ** 0.5)  # Compute attention scores, scaled
        e = F.softmax(e, dim=-1)  # Normalize scores
        h = torch.matmul(e, v1)  # Apply attention weights
        return self.intra_flow_attention_out(h)  # Return to original LSTM hidden size

    def forward(self, x):
        # x shape: (batch, time_steps, num_links)
        batch_size, time_steps, num_links = x.size()

        # Expand dimensions for CNN input
        x = x.unsqueeze(1)  # Shape: (batch, 1, time_steps, num_links)

        # Apply CNN layers with pooling
        for cnn, bn, pool, dropout in zip(self.cnn_layers, self.batch_norms, self.pooling_layers, self.dropouts):
            x = cnn(x)  # Convolution
            x = bn(x)  # Batch normalization
            x = F.relu(x)  # Activation
            x = pool(x)  # Pooling
            x = dropout(x)  # Dropout

        # Reshape for LSTM input and apply inter-flow attention
        x = x.permute(0, 2, 1, 3).contiguous()  # Rearrange to (batch, time_steps, hidden_size, reduced_links)
        x = x.view(batch_size, time_steps, -1)  # Flatten the last two dimensions for LSTM input
        x = self.inter_flow_attention(x)  # Apply inter-flow attention

        # LSTM for Temporal Modeling
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)  # Hidden state
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)  # Cell state
        lstm_out, _ = self.lstm(x, (h0, c0))

        # Apply intra-flow attention on LSTM outputs
        h = self.intra_flow_attention(lstm_out)

        # Fully Connected Layer
        out = h[:, -1, :]  # Take the last time step output
        out = self.fc(out)
        out = out.view(batch_size, 12, num_links)  # Reshape to (batch, 12, 132)
        return out


if __name__ == "__main__":

    """
    Checking compatability
    """

    # Example usage:
    directory_path = r'C:\Users\beaviv\Datasets\SNDlib\directed-abilene-zhang-5min-over-6months-ALL'
    save_directory = r'C:\Users\aviv9\Datasets\SNDlib'
    csv_filename = 'geant_traffic_matrices.csv'

    criterion = nn.MSELoss()

    # Process the XML files and save the DataFrame
    # df = process_traffic_matrices(directory_path)
    # save_dataframe_as_csv(df, save_directory, csv_filename)

    # Load the DataFrame from the saved file
    df_csv = load_dataframe_from_csv(save_directory, csv_filename)

    X_train, y_train, X_val, y_val, X_test, y_test = prepare_datasets(df=df_csv, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2)
    X_train_normalized, X_val_normalized, X_test_normalized = normalize_data(X_train, X_val, X_test, type='min_max')

    """
    RNN Predictor
    """
    input_size = len(df_csv.columns)  # Number of flows
    hidden_size = 5
    output_size = 12 * len(df_csv.columns)  # The output is flattened and will be reshaped to (12, 132). 12 for 1 hour prediction and 132 links.
    num_layers = 4
    model = RNNPredictor(input_size, hidden_size, output_size, num_layers)

    outputs = model(X_train[:32])

    loss = criterion(outputs, y_train[:32])

    """
    MLP Predictor
    """
    model = MLPPredictor(input_size, hidden_size, output_size, num_layers)

    outputs = model(X_train[:32])

    loss = criterion(outputs, y_train[:32])

    """
    GNN Predictor
    """

    A, pos = create_abilene_graph()
    G = nx.from_numpy_matrix(A, create_using=nx.DiGraph())
    data = from_networkx(G)
    edge_index = data.edge_index  # Edge index for the GCN
    # Instantiate the model
    model = GNNPredictor(input_size, hidden_size, output_size, num_layers, edge_index, num_gcn_layers=2)
    output = model(X_train[:32])
    print(output.shape)  # Expected output shape: (batch, 12, 132)

    """
    GNN + distance Predictor
    """
    # Example Parameters
    input_size = 132 + 144  # 132 links plus 12^2 distance features
    hidden_size = 64
    output_size = 12 * 132
    num_layers = 4
    num_gcn_layers = 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Instantiate the model
    edge_index = edge_index.to(device)
    model = GNNPredictorWithDistance(input_size, hidden_size, output_size, num_layers, num_gcn_layers,
                                     edge_index).to(device)

    distance = squareform(pdist(pos, metric="euclidean"))
    distance = torch.tensor(distance, dtype=torch.float)
    X_train, distance = X_train.to(device), distance.to(device)
    outputs = model(X_train[:32], distance)
    y_train = y_train.to(device)
    loss = criterion(outputs, y_train[:32])

    """
    ACRNN
    """
    # Example Parameters
    input_size = len(df_csv.columns)  # Number of flows
    hidden_size = 64
    output_size = 12 * len(df_csv.columns)
    num_layers = 4
    attention_size = 128
    dropout_prob = 0.3  # Dropout probability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate the ACRNN model
    model = ACRNN(input_size, hidden_size, output_size, num_layers, attention_size).to(device)
    output = model(X_train[:32])
    loss = criterion(output, y_train[:32])

    print('finished loading')