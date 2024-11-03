import numpy as np
import torch
import torch.nn as nn
from SNDlib_data import load_dataframe_from_csv, prepare_datasets, normalize_data, denormalize_predictions
from SNDlib_Prediction_Model import GRUPredictor, LSTMPredictor, MLPPredictor, RNNPredictor, ACRNN, GNNPredictor, GNNPredictorWithDistance
from environment.utils import create_abilene_graph, create_snd_geant_graph
import os
from Aviv_test.Queue_MDP import NetworkMDP
from diamond import DIAMOND
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import from_networkx
from scipy.spatial.distance import pdist, squareform


def get_abilene_flows():
    flows = []
    for s in range(12):
        for d in range(12):
            if s == d:
                continue
            else:
                flow = {"source": s, "destination": d, "packets": 0}
                flows.append(flow)
    return flows


def get_geant_flows():
    flows = []
    for s in range(22):
        for d in range(22):
            if s == d:
                continue
            else:
                flow = {"source": s, "destination": d, "packets": 0}
                flows.append(flow)
    return flows


def get_one_hour_prediction_matrix(model, device, X_test, y_test, index=0):

    model = model.to(device)
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient calculation
        x_test = X_test[index].to(device)
        x_test = x_test.unsqueeze(0)
        prediction = model(x_test)
        prediction = prediction.squeeze(0)
        truth = y_test[index]

    prediction, truth = prediction.cpu(), truth.cpu()

    return np.array(prediction), np.array(truth)


def DIAMOND_abilene_rate_alocation(model, dataset, device, X_test, y_test, index=0, number_of_flows=20, number_of_hours=1,  seed=123):

    np.random.seed(seed)

    index = np.random.randint(0, len(X_test) - number_of_hours)

    prediction_arrival_matrix, gt_arrival_matrix = get_one_hour_prediction_matrix(model, device, X_test, y_test,
                                                                                    index=index)
    # Stacking prediction matrices for requested hours
    for i in range(number_of_hours - 1):
        prediction_arrival_matrix = np.vstack((prediction_arrival_matrix, get_one_hour_prediction_matrix(model,device, X_test, y_test, index=index + i)[0]))
        gt_arrival_matrix = np.vstack((gt_arrival_matrix, get_one_hour_prediction_matrix(model, device, X_test, y_test,)[1]))

    prediction_arrival_matrix, gt_arrival_matrix = prediction_arrival_matrix * 100, gt_arrival_matrix * 100  # Data is in Gb we want Mb

    flow_indices = np.random.randint(0, prediction_arrival_matrix.shape[1], number_of_flows)

    prediction_arrival_matrix, gt_arrival_matrix = prediction_arrival_matrix[:, flow_indices], gt_arrival_matrix[:, flow_indices]

    if dataset == "abilene":
        new_flows = get_abilene_flows()
        graph_mode = 'abilene'
    elif dataset == "geant":
        new_flows = get_geant_flows()
        graph_mode = 'snd_geant'

    new_flows = [new_flows[idx] for idx in flow_indices]

    MODEL_PATH = os.path.join("DIAMOND", "pretrained", "model_20221113_212726_480.pt")


    alg = DIAMOND(grrl_model_path=MODEL_PATH,
                  nb3r_tmpr=1.2,
                  nb3r_steps=10
                  )

    # usage

    threshold = 1

    """
    Predicted arrival matrix
    """

    mdp = NetworkMDP(alg=alg,
                     threshold=threshold,
                     arrival_matrix=prediction_arrival_matrix,
                     packets_per_iteration=500,
                     num_nodes=5,
                     num_edges=10,
                     num_actions=4,
                     num_flows=np.shape(prediction_arrival_matrix)[1],
                     min_flow_demand=100,
                     max_flow_demand=10000,
                     min_capacity=400,
                     max_capacity=700,
                     seed=seed,
                     graph_mode=graph_mode,
                     trx_power_mode='equal',
                     rayleigh_scale=1,
                     max_trx_power=10,
                     channel_gain=1,
                     new_flows=new_flows
                     )

    (all_iterations_paths_pred,
     all_iterations_rates_pred,
     all_iterations_delays_pred) = mdp.run_episode(time_steps=len(prediction_arrival_matrix))

    # (all_iterations_paths_pred,
    #  all_iterations_rates_pred,
    #  all_iterations_rates_with_time_slicing_pred,
    #  all_iterations_delays_pred,
    #  all_iterations_delays_with_time_slicing_pred) = mdp.run_episode(
    #                                                             time_steps=len(prediction_arrival_matrix),
    #                                                             )

    # mdp.plot_rates_delays(all_iterations_rates_pred,
    #                       all_iterations_rates_with_time_slicing_pred,
    #                       all_iterations_delays_pred,
    #                       all_iterations_delays_with_time_slicing_pred)

    mdp = NetworkMDP(alg=alg,
                     threshold=threshold,
                     arrival_matrix=gt_arrival_matrix,
                     packets_per_iteration=500,
                     num_nodes=5,
                     num_edges=10,
                     num_actions=4,
                     num_flows=np.shape(gt_arrival_matrix)[1],
                     min_flow_demand=100,
                     max_flow_demand=10000,
                     min_capacity=400,
                     max_capacity=700,
                     seed=seed,
                     graph_mode=graph_mode,
                     trx_power_mode='equal',
                     rayleigh_scale=1,
                     max_trx_power=10,
                     channel_gain=1,
                     new_flows=new_flows
                     )

    (all_iterations_paths_gt,
     all_iterations_rates_gt,
     all_iterations_delays_gt) = mdp.run_episode(time_steps=len(prediction_arrival_matrix))
    # (all_iterations_paths_gt,
    #  all_iterations_rates_gt,
    #  all_iterations_rates_with_time_slicing_GT,
    #  all_iterations_delays_gt,
    #  all_iterations_delays_with_time_slicing_GT) = mdp.run_episode(
    #                                                             time_steps=len(prediction_arrival_matrix),
    #                                                             )

    # mdp.plot_rates_delays(all_iterations_rates_gt,
    #                       all_iterations_rates_with_time_slicing_GT,
    #                       all_iterations_delays_gt,
    #                       all_iterations_delays_with_time_slicing_GT)

    # return all_iterations_rates_pred, all_iterations_rates_with_time_slicing_pred, all_iterations_rates_gt, all_iterations_rates_with_time_slicing_GT
    return all_iterations_rates_pred, all_iterations_rates_gt, all_iterations_delays_pred, all_iterations_delays_gt


def main(model_name="GRU", dataset='abilene', number_of_flows=10, number_of_hours=1, seed=123):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    """
    Loading Data
    """

    # directory_path = r'C:\Users\beaviv\Datasets\SNDlib\directed-abilene-zhang-5min-over-6months-ALL'
    save_directory = r'C:\Users\beaviv\Datasets\SNDlib'
    csv_filename = f'{dataset}_traffic_matrices.csv'

    # Load the DataFrame from the saved file
    df_csv = load_dataframe_from_csv(save_directory, csv_filename)

    X_train, y_train, X_val, y_val, X_test, y_test = prepare_datasets(df=df_csv, train_ratio=0.6, val_ratio=0.3, test_ratio=0.1)

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
    Loading Model
    """
    model_path = r'C:\Users\beaviv\NTPRA\SNDlib_traffic_prediction\SNDlib_models'
    model_path = os.path.join(model_path, f'{dataset}_{model_name}_model_best_model.pk')
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))

    # prediction_matrix, truth_matrix = get_one_hour_prediction_matrix(model, device, X_test_normalized, y_test, index=0)

    # (all_iterations_rates_pred, all_iterations_rates_with_time_slicing_pred, all_iterations_rates_gt,
    #  all_iterations_rates_with_time_slicing_GT) = DIAMOND_abilene_rate_alocation(model=model,
    #                                                                              device=device,
    #                                                                              X_test=X_test,
    #                                                                              y_test=y_test,
    #                                                                              index=0,
    #                                                                              number_of_flows=number_of_flows,
    #                                                                              number_of_hours=1
    #                                                                              )



    (all_iterations_rates_pred, all_iterations_rates_gt, all_iterations_delays_pred,
     all_iterations_delays_gt) = DIAMOND_abilene_rate_alocation(model=model,
                                                                dataset=dataset,
                                                                device=device,
                                                                X_test=X_test,
                                                                y_test=y_test,
                                                                index=0,
                                                                number_of_flows=number_of_flows,
                                                                number_of_hours=number_of_hours,
                                                                seed=seed)

    print('finished')

    # return all_iterations_rates_pred, all_iterations_rates_with_time_slicing_pred, all_iterations_rates_gt, all_iterations_rates_with_time_slicing_GT
    return all_iterations_rates_pred, all_iterations_rates_gt, all_iterations_delays_pred, all_iterations_delays_gt


def plot_pred_gt(model_name='GRU', dataset='abilene', number_of_flows=[10, 20, 30], number_of_hours=1,random_trials=1,  seed=123, save_fig=False):

    rates_pred_list, rates_pred_with_tc_list, rates_gt_list, rates_gt_with_tc_list = [], [], [], []

    colors = ['blue', 'red', 'green', 'orange', 'black', 'purple']

    # fig, ax = plt.subplots(2, 1, figsize=(20, 15))  # Use axis-based plotting
    fig, ax = plt.subplots(figsize=(20, 10))  # Use axis-based plotting

    for i, flow_number in enumerate(number_of_flows):
        # (all_iterations_rates_pred, all_iterations_rates_with_time_slicing_pred, all_iterations_rates_gt,
        #  all_iterations_rates_with_time_slicing_gt) = main(model_name="GRU", number_of_flows=flow_number)

        (all_iterations_rates_pred, all_iterations_rates_gt, all_iterations_delays_pred,
         all_iterations_delays_gt) = main(model_name=model_name,
                                          dataset=dataset,
                                          number_of_flows=flow_number,
                                          number_of_hours=number_of_hours,
                                          seed=seed)

        # Rates
        mean_rates_per_time_step_pred = np.mean(all_iterations_rates_pred, axis=1)
        # mean_rates_with_time_slicing_pred = np.mean(all_iterations_rates_with_time_slicing_pred, axis=1)
        mean_rates_per_time_step_gt = np.mean(all_iterations_rates_gt, axis=1)
        # mean_rates_with_time_slicing_gt = np.mean(all_iterations_rates_with_time_slicing_gt, axis=1)

        for j in range(random_trials - 1):
            (all_iterations_rates_pred, all_iterations_rates_gt, all_iterations_delays_pred,
             all_iterations_delays_gt) = main(model_name=model_name,
                                              dataset=dataset,
                                              number_of_flows=flow_number,
                                              number_of_hours=number_of_hours,
                                              seed=seed + 10 * j)
            mean_rates_per_time_step_pred = mean_rates_per_time_step_pred + np.mean(all_iterations_rates_pred, axis=1)
            mean_rates_per_time_step_gt = mean_rates_per_time_step_gt + np.mean(all_iterations_rates_gt, axis=1)

        mean_rates_per_time_step_gt /= random_trials
        mean_rates_per_time_step_pred /= random_trials

        # delays
        mean_delays_per_time_step_pred = np.mean(all_iterations_delays_pred, axis=1)
        mean_delays_per_time_step_gt = np.mean(all_iterations_delays_gt, axis=1)

        # Rates
        ax.plot(mean_rates_per_time_step_pred, label=f'Pred {flow_number} Flows', linestyle="-", color=colors[i])
        # ax.plot(mean_rates_with_time_slicing_pred, label=f'no Demand Slicing Pred {flow_number} Flows', linestyle="-", color=colors[2*i + 1], marker='*')
        ax.plot(mean_rates_per_time_step_gt, label=f'GT {flow_number} Flows', linestyle="--", color=colors[i])
        # ax.plot(mean_rates_with_time_slicing_gt, label=f'no Demand Slicing GT {flow_number} Flows', linestyle="--", color=colors[2*i + 1], marker='*')

        # Delays
        # ax[1].plot(mean_delays_per_time_step_pred, label=f'Pred {flow_number} Flows', linestyle="-", color=colors[2*i + 1])
        # ax[1].plot(mean_delays_per_time_step_gt, label=f'GT {flow_number} Flows', linestyle="--", color=colors[2*i + 1])


        # rates_pred_list.append(all_iterations_rates_pred)
        # rates_pred_with_tc_list.append(all_iterations_rates_with_time_slicing_pred)
        # rates_gt_list.append(all_iterations_rates_gt)
        # rates_gt_with_tc_list.append(all_iterations_rates_with_time_slicing_gt)

        # plt.pause(0.001)  # Force update of the plot after each iteration

    # Rates
    ax.legend(loc='best')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Rate[Mbps]')
    ax.set_title(f"Mean Rate Allocation {model_name} Model for {dataset} Flows Prediction vs GT")
    ax.grid(True)

    # Delays
    # ax[1].legend(loc='best')
    # ax[1].set_xlabel('Time Steps')
    # ax[1].set_ylabel('Delays')
    # ax[1].set_title("Delay for Abilene Flows Prediction vs GT")
    # ax[1].grid(True)

    # Ensure the directory exists
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Traffic_prediction_loss_curves')
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f'Rate_allocation_Pred_GT_{model_name}_{dataset}.png')
    if save_fig:
        plt.savefig(save_path)
    plt.show()
    plt.close()


if __name__ == '__main__':
    models = ["MLP"]

    for model in models:
        plot_pred_gt(model_name=model, dataset='geant', number_of_flows=[10, 20, 30], number_of_hours=10, random_trials=5, seed=123, save_fig=True)

