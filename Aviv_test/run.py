from utils import transmission_slicing
from environment.data import generate_env
from diamond import DIAMOND
import os
import numpy as np


def get_paths(num_flows=10, num_nodes=20, num_edges=30, num_actions=4, temperature=1.2,
              nb3r_steps=100, min_flow_demand=5000, max_flow_demand=5500,
              min_capacity=100, max_capacity=600, seed=223, trx_power_mode='equal',
              rayleigh_scale=1, max_trx_power=10, channel_gain=1,
              packets_per_iteration=1e3, GRAPH_MODE='random'):

    MODEL_PATH = os.path.join("DIAMOND", "pretrained", "model_20221113_212726_480.pt")
    alg = DIAMOND(grrl_model_path=MODEL_PATH, nb3r_tmpr=temperature, nb3r_steps=nb3r_steps)

    env = generate_env(num_nodes=num_nodes, num_edges=num_edges, num_actions=num_actions,
                       num_flows=num_flows, min_flow_demand=min_flow_demand,
                       max_flow_demand=max_flow_demand, min_capacity=min_capacity,
                       max_capacity=max_capacity, seed=seed, graph_mode=GRAPH_MODE,
                       trx_power_mode=trx_power_mode, rayleigh_scale=rayleigh_scale,
                       max_trx_power=max_trx_power, channel_gain=channel_gain)

    # env.show_graph()
    all_iterations_paths = transmission_slicing(env=env,
                                                alg=alg,
                                                num_actions=num_actions,
                                                packets_per_iteration=packets_per_iteration,
                                                min_flow_demand=min_flow_demand,
                                                max_flow_demand=max_flow_demand,
                                                min_capacity=min_capacity,
                                                max_capacity=max_capacity,
                                                seed=seed,
                                                graph_mode=GRAPH_MODE,
                                                trx_power_mode=trx_power_mode,
                                                rayleigh_scale=rayleigh_scale,
                                                max_trx_power=max_trx_power,
                                                channel_gain=channel_gain,
                                                return_paths=True,
                                                num_time_steps=60)

    return all_iterations_paths


if __name__ == "__main__":

    paths = get_paths()

    print("Got them")















