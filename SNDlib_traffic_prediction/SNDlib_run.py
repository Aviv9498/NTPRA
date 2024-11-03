import torch
import torch.nn as nn
from SNDlib_data import load_dataframe_from_csv, prepare_datasets, normalize_data
from SNDlib_Prediction_Model import GRUPredictor, RNNPredictor, MLPPredictor, LSTMPredictor, GNNPredictor, GNNPredictorWithDistance, ACRNN
from SNDlib_train import train,plot_curves, test_model, MAE_MAPE_MSE_test, plot_MAE_bar, plot_mae_mse_bar
import os
from environment.utils import create_abilene_graph, create_snd_geant_graph
import networkx as nx
from torch_geometric.data import Data, Batch
from torch_geometric.utils import from_networkx
from scipy.spatial.distance import pdist, squareform


def run(model_name="GRU", dataset='abilene', training=False, test=False):

    """
    @param dataset: which dataset we are working on
    @param model_name: chose Model
    @param training: If training run
    @param test: If testing run
    @return: training and testing  model
    """

    """
    Loading Data
    """

    # directory_path = r'C:\Users\beaviv\Datasets\SNDlib\directed-abilene-zhang-5min-over-6months-ALL'
    save_directory = r'C:\Users\beaviv\Datasets\SNDlib'
    # save_directory = r'C:\Users\aviv9\Datasets\SNDlib'
    csv_filename = f'{dataset}_traffic_matrices.csv'

    # Load the DataFrame from the saved file
    df_csv = load_dataframe_from_csv(save_directory, csv_filename)

    # For Drive operation
    # df_csv = pandas.read_csv(drive_path+ f'{dataset}_traffic_matrices.csv')
    # print(drive_path+ f'{dataset}_traffic_matrices.csv')

    if dataset == 'geant':
      df_csv = df_csv.drop(df_csv.index[5136:5148])

    X_train, y_train, X_val, y_val, X_test, y_test = prepare_datasets(df=df_csv, train_ratio=0.6, val_ratio=0.1, test_ratio=0.3)

    X_train_normalized, X_val_normalized, X_test_normalized, y_train_normalized, y_val_normalized, y_test_normalized = normalize_data(X_train, X_val, X_test, y_train, y_val, y_test, type='min_max')

    model_parameters = {'input_size': len(df_csv.columns),  # Number of links
                        'hidden_size': 128,
                        'output_size': 12 * len(df_csv.columns),  # The output is flattened and will be reshaped to (12, 132). 12 for 1 hour prediction and 132 links.
                        'num_layers': 2,
                        }

    # Model Selection
    if model_name == 'GRU':
        model = GRUPredictor(**model_parameters)
    elif model_name == 'LSTM':
        model = LSTMPredictor(**model_parameters)
    elif model_name == 'RNN':
        model = RNNPredictor(**model_parameters)
    elif model_name == 'MLP':
        model = MLPPredictor(**model_parameters)
    elif model_name == 'GNN':
        if dataset == "abilene":
            A, pos = create_abilene_graph()
        elif dataset == 'geant':
            A, pos = create_snd_geant_graph()

        G = nx.from_numpy_array(A, create_using=nx.DiGraph())
        data = from_networkx(G)
        model_parameters["edge_index"] = data.edge_index.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))  # Edge index for the GCN
        model_parameters["num_gcn_layers"] = 3
        model = GNNPredictor(**model_parameters)
    elif model_name == 'GNNWithDistance':
        A, pos = create_abilene_graph()
        G = nx.from_numpy_matrix(A, create_using=nx.DiGraph())
        data = from_networkx(G)
        model_parameters["edge_index"] = data.edge_index.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))  # Edge index for the GCN
        model_parameters["num_gcn_layers"] = 3
        model_parameters["input_size"] += 144
        model = GNNPredictorWithDistance(**model_parameters)
        distance = squareform(pdist(pos, metric="euclidean"))
        distance = torch.tensor(distance, dtype=torch.float).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    elif model_name == 'ACRNN':
        model_parameters["attention_size"] = 128
        model = ACRNN(**model_parameters)

    """
    Test and generate prediction
    """
    if test:
        print(f'Testing {model_name} Model on {dataset} Dataset')

        # PC path
        model_path = r'C:\Users\beaviv\DIAMOND\SNDlib_traffic_prediction\SNDlib_models'

        # Laptop path
        # model_path = r'C:\Users\aviv9\DIAMOND\SNDlib_traffic_prediction\SNDlib_models'

        # Drive path
        # model_path = '/content/gdrive/My Drive/SNDlib/SNDlib_models/'

        model_path = os.path.join(model_path, f'{dataset}_{model_name}_model_best_model.pk')
        # model_path = os.path.join(model_path, f'model_280.pk')
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))

        # Only for this model need an extra argument:distance map
        if isinstance(model, GNNPredictorWithDistance):
            mean_mae, mean_mape, mean_mse = MAE_MAPE_MSE_test(model, X_test, y_test,
                                                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                                                distance=distance)
        else:
            mean_mae, mean_mape, mean_mse = MAE_MAPE_MSE_test(model, X_test, y_test
                                                              , device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        return mean_mae, mean_mape, mean_mse

    """
    Train
    """
    if training:

        # Hyperparameters
        train_hyperparameters = {'model': model,
                                 'model_name': model_name,
                                 'dataset': dataset,
                                 'X_train': X_train,
                                 'y_train': y_train,
                                 'X_val': X_val,
                                 'y_val': y_val,
                                 'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                                 'optimizer': torch.optim.Adam,
                                 'criterion': nn.L1Loss(),  # nn.MSELoss()
                                 'num_epochs': 300,
                                 'batch_size': 32,
                                 'lr': 1e-3,
                                 'print_every': 50,
                                 'save_every': 3000,
                                 'weight_decay': 1e-3,
                                 'l1_parameter': 0,
                                 'load_model': False,
                                 'WANDB_TRACKING': True,
                                 'distance': None,
                                 'use_schedualer':True
                                 }

        if isinstance(model, GNNPredictorWithDistance):
            train_hyperparameters['distance'] = distance.to(train_hyperparameters['device'])

        train_loss, validation_loss, best_model = train(**train_hyperparameters)
        # plot_curves(train_losses=train_loss, val_losses=validation_loss, model_name=model_name, dataset=dataset)


if __name__ == "__main__":

    mae_values = []
    mse_values = []

    models = ["GNN"]  # "MLP", "RNN", "GRU", "LSTM", "GNN",
    for model_name in models:
        run(model_name=model_name, dataset='geant', training=True, test=False)

        # mean_mae, mean_mape, mean_mse = run(model_name=model_name, dataset='geant', training=False, test=True)
        # mae_values.append(mean_mae)
        # mse_values.append(mean_mse)

    # plot_mae_mse_bar(models=models, mae_values=mae_values, mse_values=mse_values, dataset='geant')

    if torch.cuda.is_available():
      torch.cuda.empty_cache()  # Clear the CUDA cache
      torch.cuda.synchronize()  # Wait for all streams on GPU to finish
    print('Done')
