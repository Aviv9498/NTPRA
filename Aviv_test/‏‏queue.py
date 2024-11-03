import numpy as np
import matplotlib.pyplot as plt
from environment import generate_env
from diamond import DIAMOND
import os
import random
import copy


def simulate_all_poisson_arrivals(num_queues, lambdas, time_steps):
    """
    Simulate Poisson arrivals for multiple queues over a number of time steps.

    Parameters:
    - num_queues: Number of queues.
    - lambdas: List of arrival rates for each queue.
    - time_steps: Number of time steps to simulate.

    Returns:
    - all_arrivals: List of arrays, each array containing number of arrivals per time step for each queue.
    """
    all_arrivals = np.zeros((num_queues, time_steps))

    for i in range(num_queues):
        arrivals = np.random.poisson(lambdas[i], time_steps)
        all_arrivals[i, :] = arrivals

    return all_arrivals


def get_random_flows(num_nodes, num_flows, demands, seed=1):
    """
    generates random flows
    :param num_nodes: number of nodes in the communication graph
    :param num_flows: number of flows in the communication graph
    :param demands: list of packets demands for flows to choose from
    :param seed: random seed
    :return: list of flows as (src, dst, pkt)
    """
    random.seed(seed)
    result = []
    for i in range(num_flows):
        src, dst = random.sample(range(num_nodes), 2)
        f = {"source": src,
             "destination": dst,
             "packets": demands[i]}
        result.append(f)
    return result


def queue_theory_paths(num_flows=10, num_nodes=20, num_edges=30, num_actions=4, temperature=1.2,
                       nb3r_steps=100, min_flow_demand=5000, max_flow_demand=5500,
                       min_capacity=100, max_capacity=600, seed=223, trx_power_mode='equal',
                       rayleigh_scale=1, max_trx_power=10, channel_gain=1, GRAPH_MODE='random',
                       boarder_line=100, time_steps=60, lambdas=np.arange(1, 11)
                       ):

    np.random.seed(seed)
    MODEL_PATH = os.path.join("DIAMOND", "pretrained", "model_20221113_212726_480.pt")

    alg = DIAMOND(grrl_model_path=MODEL_PATH,
                  nb3r_tmpr=temperature,
                  nb3r_steps=nb3r_steps
                  )

    # Each flow's Queue gets different lambda for poisson arrival rate
    poisson_packets = simulate_all_poisson_arrivals(num_queues=num_flows,
                                                    lambdas=lambdas,
                                                    time_steps=time_steps
                                                    )

    # Initial packets are the poisson_packets from first time steps
    flows = get_random_flows(num_nodes=num_nodes,
                             num_flows=num_flows,
                             demands=poisson_packets[:, 0])

    env = generate_env(num_nodes=num_nodes,
                       num_edges=num_edges,
                       num_actions=num_actions,
                       num_flows=num_flows,
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
                       given_flows=flows
                       )

    """
    initializing variables
    """

    # TODO : possible that not all flows are alive at the start
    alive_flows_indices = np.array(list(range(env.num_flows)))

    current_demand = copy.copy(env.packets)

    new_flows = copy.deepcopy(env.flows)

    all_iterations_paths = [[] for _ in range(env.num_flows)]

    all_iteration_rates = np.zeros((time_steps, env.num_flows))  # [[] for _ in range(env.num_flows)]

    """
    start time step loop
    """
    for timestep in range(time_steps):

        env = generate_env(num_nodes=num_nodes,
                           num_edges=num_edges,
                           num_actions=num_actions,
                           num_flows=num_flows,
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
                           given_flows=new_flows
                           )

        diamond_paths = alg(env=env, use_nb3r=True)

        # Adding paths
        for idx, route in zip(alive_flows_indices, diamond_paths):
            all_iterations_paths[idx].append(route)

        # For  dead flows we give the same path
        # TODO : need to change
        for flow_idx, path_list in enumerate(all_iterations_paths):
            if len(path_list) != timestep + 1:
                all_iterations_paths[flow_idx].append(all_iterations_paths[flow_idx][-1])

        # calculating rates
        flows_rate_per_iter = env.flows_rate

        # Adding rates
        all_iteration_rates[timestep, :] = flows_rate_per_iter

        # Updating current demand
        current_demand = np.array(current_demand) - flows_rate_per_iter
        # adding arrival packets
        current_demand = np.array(current_demand) + poisson_packets[:, timestep + 1]  # + 1 because first demand is col 0

        new_flows = [flow for flow, left_demand in zip(new_flows, current_demand) if left_demand > boarder_line]

        # Updating remaining flows indices
        alive_flows_indices = np.nonzero(current_demand > boarder_line)[0]









    return all_iterations_paths













if __name__ == "__main__":

    paths = queue_theory_paths(num_flows=5, lambdas=[100, 150, 200, 250, 300], time_steps=10)

    print("Got them")
