import copy
import random
import time
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from environment import generate_env
from diamond import DIAMOND
import os
import networkx as nx
import matplotlib.colors as mcolors
from scipy.stats import levy_stable, pareto, lognorm


def transmission_slicing(env, alg, packets_per_iteration, return_paths=False, num_time_steps=60, **kwargs):
    """
    :param env: env after setting real demands
    :param alg: DIAMOND alg
    :param packets_per_iteration: no. of packets we stream every iter for comparison
    :para return_paths: return the paths in each iteration
    """
    counter_aviv = env.num_flows  # For Debugging
    counter_raz = env.num_flows

    paths_with_full_demand, rl_path_idx_out_of_4 = alg(env, use_nb3r=False, return_rl_actions=True)

    # We keep the actions for grrl.run() if we want to specify routes
    rl_actions_full_demand = [[q, path_idx] for q, path_idx in enumerate(rl_path_idx_out_of_4)]

    test_paths = alg(env, use_nb3r=False, given_actions=rl_actions_full_demand)

    rates_for_full_demand = env.flows_rate
    delay_for_full_demand = env.flows_delay

    # current demands are different because rates are different. demand = demand - rate_for_iteration
    current_demand_aviv = copy.copy(env.packets)
    current_demand_raz = copy.copy(env.packets)

    # k = math.ceil(env.max_demand / packets_per_iteration)  # we run until the flow with max demand finishes

    #  Paths, Rates and Delay arrays Initialization, Aviv/Raz for comparison

    # Aviv
    all_iteration_rates_aviv = [[] for _ in range(env.num_flows)]  # np.zeros((k, env.num_flows))
    all_iteration_delay_aviv = [[] for _ in range(env.num_flows)]  # np.zeros((k, env.num_flows))
    all_iterations_average_rate_aviv = []
    all_iterations_average_delay_aviv = []

    # Raz
    all_iteration_rates_raz = [[] for _ in range(env.num_flows)]  # np.zeros((k, env.num_flows))
    all_iteration_delay_raz = [[] for _ in range(env.num_flows)]  # np.zeros((k, env.num_flows))
    all_iterations_average_rate_raz = []
    all_iterations_average_delay_raz = []

    all_possible_paths_aviv = [[] for _ in range(env.num_flows)]
    all_possible_paths_raz = [[] for _ in range(env.num_flows)]

    # Initializing "new flows" dict to work with every iteration, different flows are "alive" for aviv\raz
    new_flows_aviv = copy.deepcopy(env.flows)
    new_flows_raz = copy.deepcopy(env.flows)

    # Mapping flows to indices

    remaining_flows_indices_aviv = np.array(list(range(env.num_flows)))
    remaining_flows_indices_raz = np.array(list(range(env.num_flows)))

    # inserting demand for new flows, the demand will stay for all iters
    for flow1, flow2 in zip(new_flows_aviv, new_flows_raz):
        flow1["packets"] = packets_per_iteration
        flow2["packets"] = packets_per_iteration

    print_once = True
    save_once_aviv = True
    save_once_raz = True
    # Starting main loop

    aviv_finished = False
    raz_finished = False
    i = 0  # Counter for placements in global lists
    # for i in range(k):
    while (not aviv_finished) or (not raz_finished):

        # If we run more than asked stop
        if i >= num_time_steps:
            break

        """
        Aviv
        """
        if not aviv_finished:

            # Setting env
            env_aviv = generate_env(num_nodes=env.num_nodes,
                                num_edges=env.num_edges//2,  # env.num edges = 2 * num edges specified in generate env
                                num_actions=env.k,
                                num_flows=env.num_flows,
                                min_flow_demand=kwargs.get('min_flow_demand', 100),
                                max_flow_demand=kwargs.get('max_flow_demand', 200),
                                min_capacity=kwargs.get('min_capacity', 200),
                                max_capacity=kwargs.get('max_capacity', 500),
                                seed=kwargs.get('seed', 123),
                                graph_mode=kwargs.get('graph_mode', 'random'),
                                trx_power_mode=kwargs.get('trx_power_mode', 'equal'),
                                rayleigh_scale=kwargs.get('rayleigh_scale', 1),
                                max_trx_power=kwargs.get('max_trx_power', 10),
                                channel_gain=kwargs.get('channel_gain', 1),
                                given_flows=new_flows_aviv)

            # Aviv
            diamond_paths_aviv = alg(env_aviv, use_nb3r=False)
            flows_rate_per_iter_aviv = env_aviv.flows_rate
            flows_delay_per_iter_aviv = env_aviv.flows_delay

            # For debugging
            if len(remaining_flows_indices_aviv) != counter_aviv:
                # few can finish in same iteration so not good for tracking
                counter_aviv -= 1
                # print("one more fell aviv")

            """
            Updating Rates, Delay arrays
            """

            # rates is a list of remaining flows rate given in  the last iteration
            if i == 0:
                # Aviv
                rates_aviv = flows_rate_per_iter_aviv
                delays_aviv = flows_delay_per_iter_aviv

            # Aviv
            rates_aviv[np.array(remaining_flows_indices_aviv)] = flows_rate_per_iter_aviv
            delays_aviv[np.array(remaining_flows_indices_aviv)] = flows_delay_per_iter_aviv

            # Adding average to the list
            all_iterations_average_delay_aviv.append(np.mean(delays_aviv[delays_aviv != 0]))
            all_iterations_average_rate_aviv.append(np.mean(rates_aviv[rates_aviv != 0]))

            # all_iteration_rates_aviv[i].append(list(rates_aviv))  # [i] = rates_aviv
            # all_iteration_delay_aviv[i].append(list(delays_aviv))  # [i] = delays_aviv
            for flow_id in range(env.num_flows):
                all_iteration_rates_aviv[flow_id].append(rates_aviv[flow_id])
                all_iteration_delay_aviv[flow_id].append(delays_aviv[flow_id])

            """
            Adding current paths to paths list
            """
            # Aviv
            for idx, route in zip(remaining_flows_indices_aviv, diamond_paths_aviv):
                all_possible_paths_aviv[idx].append(route)

            # For flows that finished we give the same path
            for flow_idx, path_list in enumerate(all_possible_paths_aviv):
                if len(path_list) != i + 1:
                    all_possible_paths_aviv[flow_idx].append(all_possible_paths_aviv[flow_idx][-1])

            # Updating current demand, every iter we subtract transmitted packets from og demand

            current_demand_aviv = np.array(current_demand_aviv) - rates_aviv

            # checking remaining flows
            remaining_flows_demands = current_demand_aviv[np.array(remaining_flows_indices_aviv)]

            # TODO: think here. for now "alive" flows have same size of packets every iter.
            # for id, flow in enumerate(new_flows_aviv):
            #     flow["packets"] = min(flow["packets"], remaining_flows_demands[id])

            new_flows_aviv = [flow for flow, left_demand in zip(env_aviv.flows, remaining_flows_demands) if left_demand > 0]


            """
            for flows that finished transmitting we give a rate delay of 0
            """

            # Aviv
            rates_aviv = np.array([rate if current_demand_aviv[i] > 0 else 0 for i, rate in enumerate(rates_aviv)])
            delays_aviv = np.array([delay if current_demand_aviv[i] > 0 else 0 for i, delay in enumerate(delays_aviv)])

            # Updating remaining flows indices
            remaining_flows_indices_aviv = np.nonzero(current_demand_aviv > 0)[0]
            num_fallen_flows = env.num_flows - len(remaining_flows_indices_aviv)

            # Checking if there are Flows left
            if len(remaining_flows_indices_aviv) == 0:
                aviv_finished = True

        """
        Raz
        """
        if not raz_finished:

            # Setting env
            env_raz = generate_env(num_nodes=env.num_nodes,
                                    num_edges=env.num_edges // 2,  # env.num edges = 2 * num edges specified in generate env
                                    num_actions=env.k,
                                    num_flows=env.num_flows,
                                    min_flow_demand=kwargs.get('min_flow_demand', 100),
                                    max_flow_demand=kwargs.get('max_flow_demand', 200),
                                    min_capacity=kwargs.get('min_capacity', 200),
                                    max_capacity=kwargs.get('max_capacity', 500),
                                    seed=kwargs.get('seed', 123),
                                    graph_mode=kwargs.get('graph_mode', 'random'),
                                    trx_power_mode=kwargs.get('trx_power_mode', 'equal'),
                                    rayleigh_scale=kwargs.get('rayleigh_scale', 1),
                                    max_trx_power=kwargs.get('max_trx_power', 10),
                                    channel_gain=kwargs.get('channel_gain', 1),
                                    given_flows=new_flows_raz)

            if len(remaining_flows_indices_raz) != counter_raz:
                counter_raz -= 1
                # print("one more fell raz")

            # Taking the original path for remaining flows
            remaining_rl_actions_full_demand = [[current_flows_idx, rl_actions_full_demand[original_flow_idx][1]]
                                                for current_flows_idx, original_flow_idx in
                                                enumerate(remaining_flows_indices_raz)]

            diamond_paths_raz = alg(env_raz, use_nb3r=False, given_actions=remaining_rl_actions_full_demand)
            flows_rate_per_iter_raz = env_raz.flows_rate
            flows_delay_per_iter_raz = env_raz.flows_delay

            """
            Updating Rates, Delay arrays
            """

            # rates is a list of remaining flows rate given in  the last iteration
            if i == 0:
                # Raz
                rates_raz = flows_rate_per_iter_raz
                delays_raz = flows_delay_per_iter_raz

            # Raz
            rates_raz[np.array(remaining_flows_indices_raz)] = flows_rate_per_iter_raz
            delays_raz[np.array(remaining_flows_indices_raz)] = flows_delay_per_iter_raz

            # Adding averages to lists
            all_iterations_average_delay_raz.append(np.mean(delays_raz[delays_raz != 0]))
            all_iterations_average_rate_raz.append(np.mean(rates_raz[rates_raz != 0]))

            # all_iteration_rates_raz[i].append(list(rates_aviv))  # [i] = rates_aviv
            # all_iteration_delay_raz[i].append(list(delays_aviv))  # [i] = delays_aviv
            for flow_id in range(env.num_flows):
                all_iteration_rates_raz[flow_id].append(rates_raz[flow_id])
                all_iteration_delay_raz[flow_id].append(delays_raz[flow_id])

            """
            Adding current paths to paths list
            """

            # Raz
            for idx, route in zip(remaining_flows_indices_raz, diamond_paths_raz):
                all_possible_paths_raz[idx].append(route)
            for flow_idx, path_list in enumerate(all_possible_paths_raz):
                if len(path_list) != i + 1:
                    all_possible_paths_raz[flow_idx].append(all_possible_paths_raz[flow_idx][-1])

            # Updating current demand, every iter we subtract transmitted packets from og demand

            # current_demand_aviv = np.array(current_demand_aviv) - packets_per_iteration * np.ones_like(current_demand_aviv)

            current_demand_raz = np.array(current_demand_raz) - rates_raz

            # checking remaining flows
            remaining_flows_demands = current_demand_raz[np.array(remaining_flows_indices_raz)]
            new_flows_raz = [flow for flow, left_demand in zip(env_raz.flows, remaining_flows_demands) if
                             left_demand > 0]

            """
            for flows that finished transmitting we give a rate delay of 0
            """
            # Raz
            rates_raz = np.array([rate if current_demand_raz[i] > 0 else 0 for i, rate in enumerate(rates_raz)])
            delays_raz = np.array([delay if current_demand_raz[i] > 0 else 0 for i, delay in enumerate(delays_raz)])

            # Updating remaining flows indices
            remaining_flows_indices_raz = np.nonzero(current_demand_raz > 0)[0]

            # Checking if there are Flows left
            """
            If its not a simulation no need to calc all for raz just ours
            """
            if len(remaining_flows_indices_raz) or return_paths == 0:
                raz_finished = True

        # For possible break and debugging
        if all_iterations_average_rate_aviv[-1] != all_iterations_average_rate_raz[-1] and print_once:
            # print(f"first jump at iteration {i}, Aviv:{all_iterations_average_rate_aviv[-1]}, Raz:{all_iterations_average_rate_raz[-1]}")
            # print(f'number of flows that fell : {env.num_flows - env_aviv.num_flows} / {env.num_flows}')
            print_once = False

        if env_aviv.num_flows <= env.num_flows // 2 and save_once_aviv:
            half_flows_fell_index_aviv = i
            half_flows_fell_rate_aviv = all_iterations_average_rate_aviv[-1]
            save_once_aviv = False
            print(f"rate with half aviv {all_iterations_average_rate_aviv[-1]}")

        if env_raz.num_flows <= env.num_flows // 2 and save_once_raz:
            half_flows_fell_index_raz = i
            half_flows_fell_rate_raz = all_iterations_average_rate_raz[-1]
            save_once_raz = False
            print(f"rate with half raz {all_iterations_average_rate_raz[-1]}")

        # If we just need to run until half flows finished
        # Just for simulation, we need to return paths don't enter
        if (not save_once_aviv) and (not save_once_raz) and (not return_paths):
            break

        i += 1

    # Converting to np arrays
    all_iterations_average_rate_aviv = np.array(all_iterations_average_rate_aviv)
    all_iterations_average_delay_aviv = np.array(all_iterations_average_delay_aviv)
    all_iterations_average_rate_raz = np.array(all_iterations_average_rate_raz)
    all_iterations_average_delay_raz = np.array(all_iterations_average_delay_raz)

    if return_paths:
        return all_possible_paths_aviv
    else:
        return all_iterations_average_rate_aviv, all_iterations_average_delay_aviv, all_iterations_average_rate_raz, all_iterations_average_delay_raz, half_flows_fell_index_aviv, half_flows_fell_index_raz


def first_change(all_iterations_average_rate_aviv, all_iterations_average_rate_raz, all_iterations_average_delay_aviv,
                 all_iterations_average_delay_raz):

    """
    We want to check how the Rate - Delays changes at the first time they are not the same
    """

    min_length = min(len(all_iterations_average_rate_aviv), len(all_iterations_average_rate_raz))

    # Rate
    first_change_index_rate = np.argmax(np.not_equal(all_iterations_average_rate_aviv[:min_length], all_iterations_average_rate_raz[:min_length]))
    first_change_rate = [all_iterations_average_rate_aviv[first_change_index_rate], all_iterations_average_rate_raz[first_change_index_rate]]
    before_change_rates = all_iterations_average_rate_aviv[0]  # first_change_index_rate - 1
    first_change_rate_difference = first_change_rate[0] - first_change_rate[1]

    # Delay
    first_change_index_delay = np.argmax(np.not_equal(all_iterations_average_delay_aviv[:min_length], all_iterations_average_delay_raz[:min_length]))
    first_change_delay = [all_iterations_average_delay_aviv[first_change_index_delay], all_iterations_average_delay_raz[first_change_index_delay]]
    before_change_delays = all_iterations_average_delay_aviv[0]  # first_change_index_delay - 1
    first_change_delay_difference = first_change_delay[0] - first_change_delay[1]

    return first_change_index_rate, first_change_index_delay, first_change_rate, first_change_delay, before_change_rates, before_change_delays


def time_cutting(env, alg, k=5, **kwargs) -> object:
    """
    :param env: env after setting real demands
    :param alg: DIAMOND alg
    :param k: the number of time we decide a make next allocation times 1 second.
    if we decide as example allocation every 1 minute, then k=60
    """
    # diamond_real_paths = alg(env)
    # diamond_real_capacities = calc_rate_with_path(env, diamond_real_paths)
    real_demands = env.packets.copy()  # original demand received
    new_flows = copy.deepcopy(env.flows)
    original_flows = copy.deepcopy(env.flows)
    seed = env.seed

    # setting very large demand to make sure that the flow rate will in regard to the bottleneck
    # ** not the real demand **
    for i, flow in enumerate(new_flows):
        flow["packets"] = new_flows[i]["packets"] + k * env.max_capacity

    # Mapping flows to indices
    flow_index_mapping = {tuple(flow.items()): index for index, flow in enumerate(new_flows)}

    current_demand = real_demands

    all_iterations_rates = np.zeros((k, env.num_flows))
    all_iterations_capacities = np.zeros((k, env.num_flows))
    all_iterations_capacities_same_route = np.zeros((k, env.num_flows))
    # all possible paths
    all_possible_paths = [[] for _ in range(env.num_flows)]

    rates_average = []
    deltas_average = []
    deltas_average_2 = []

    for j in range(k):

        # Creating env to be able to run DIAMOND and get rates
        # supposed to be the same env , with\without flows that finish transmitting
        env2 = generate_env(num_nodes=env.num_nodes,
                            num_edges=env.num_edges//2,  # env.num edges = 2 * num edges specified in generate env
                            num_actions=env.k,
                            num_flows=env.num_flows,
                            min_flow_demand=kwargs.get('min_flow_demand', 100),
                            max_flow_demand=kwargs.get('max_flow_demand', 200),
                            min_capacity=kwargs.get('min_capacity', 200),
                            max_capacity=kwargs.get('max_capacity', 500),
                            seed=seed,
                            graph_mode=kwargs.get('graph_mode', 'random'),
                            trx_power_mode=kwargs.get('trx_power_mode', 'equal'),
                            rayleigh_scale=kwargs.get('rayleigh_scale', 1),
                            max_trx_power=kwargs.get('max_trx_power', 10),
                            channel_gain=kwargs.get('channel_gain', 1),
                            given_flows=new_flows)

        # getting paths, updated env for large demand
        # making sure flows rate follow the bottleneck
        diamond_paths, grrl_rates_data, grrl_delay_data = alg(env2, grrl_data=True)

        flows_capacities_with_interference = np.array(calc_rate_with_path(env2, diamond_paths))

        delay_data = env2.get_delay_data()
        rates_data = env2.get_rates_data()

        rates_average.append(env2.get_rates_data()["sum_flow_rates"] / env2.num_flows)
        """
        Updating delay and rates
        """

        if j == 0:
            original_rates = rates_data["rate_per_flow"]
            all_iterations_rates[j] = original_rates

            original_capacities_with_interference = flows_capacities_with_interference
            all_iterations_capacities[j] = original_capacities_with_interference

            original_capacities_same_route = flows_capacities_with_interference
            all_iterations_capacities_same_route[j] = original_capacities_same_route

        if j != 0:

            # indices of remaining flows in the original flows list
            indices = [flow_index_mapping.get(tuple(desired_flow.items()), None) for desired_flow in new_flows]
            original_rates[np.array(indices)] = rates_data["rate_per_flow"]
            all_iterations_rates[j] = original_rates

            # Calculating delta
            nonzero_indices = np.nonzero(original_rates)
            delta = (all_iterations_rates[j])[nonzero_indices] - (all_iterations_rates[j-1])[nonzero_indices]
            deltas_average.append(np.mean(delta))

            original_capacities_with_interference[np.array(indices)] = flows_capacities_with_interference
            all_iterations_capacities[j] = original_capacities_with_interference

            # Calculating delta for capacities
            delta_2 = (all_iterations_capacities[j])[nonzero_indices] - (all_iterations_capacities[j-1])[nonzero_indices]
            deltas_average_2.append(np.mean(delta_2))

        """
        adding paths for the vector of chosen paths for each flow
        """

        remaining_flows_indices = [flow_index_mapping.get(tuple(desired_flow.items()), None) for desired_flow in
                                   env2.flows]

        for idx, route in zip(remaining_flows_indices, diamond_paths):
            all_possible_paths[idx].append(route)
        for flow_idx, path_list in enumerate(all_possible_paths):
            if len(path_list) != j + 1:
                all_possible_paths[flow_idx].append(all_possible_paths[flow_idx][-1])

        """
                if j == 0:
            all_possible_paths = copy.deepcopy(diamond_paths)
        else:
            for index_of_remaining_flow, path in zip(remaining_flows_indices, diamond_paths):
                if j == 1:
                    all_possible_paths[index_of_remaining_flow] = ([all_possible_paths[index_of_remaining_flow]]
                                                                   + [path])
                if j > 1:
                    all_possible_paths[index_of_remaining_flow] += [path]
            # for all finished flows the path is the same as last allocation
            for i, flow_paths in enumerate(all_possible_paths):
                if len(flow_paths) != j + 1:
                    if j == 1:
                        all_possible_paths[i] = [all_possible_paths[i]] + [all_possible_paths[i]]
                    if j > 1:
                        all_possible_paths[i] += [all_possible_paths[i][-1]]
        """


        # Calculating capacities my way with the original route
        if j >= 1:
            paths = [all_possible_paths[m][0] for m in indices]
            capacities_with_original_route = calc_rate_with_path(env2, paths=paths)
            original_capacities_same_route[np.array(indices)] = capacities_with_original_route
            original_capacities_same_route = np.array([rate if current_demand[i] > 0 else 0 for i, rate in enumerate(original_capacities_same_route)])
            all_iterations_capacities_same_route[j] = original_capacities_same_route

        # Updating demands for remaining flows
        current_demand = np.array(current_demand) - np.array(original_rates)

        # flows that didn't finish transmitting after last iteration
        remaining_flows_indices = [flow_index_mapping.get(tuple(desired_flow.items()), None) for desired_flow in env2.flows]
        remaining_flows_demands = current_demand[np.array(remaining_flows_indices)]
        new_flows = [flow for flow, demand in zip(env2.flows, remaining_flows_demands) if demand > 0]
        # new_flows = [flow for i, flow in enumerate(env2.flows) if current_demand[i] > 0]

        # for flows that finished transmitting we give a rate of 0
        original_rates = np.array([rate if current_demand[i] > 0 else 0 for i, rate in enumerate(original_rates)])
        original_capacities_with_interference = np.array([capacity if current_demand[i] > 0 else 0 for i, capacity in
                                                          enumerate(original_capacities_with_interference)])

        #  in case all flows finished transmitting
        #  we need to give the same path as the last allocation for each flow
        if len(new_flows) == 0:
            if j == 0:
                all_possible_paths = [[path] * k for path in all_possible_paths]
            else:
                all_possible_paths = [paths + [paths[-1]] * (k - 1 - j) for paths in all_possible_paths]

            for i in range(k - 1 - j):
                all_iterations_rates[j + i + 1] = np.zeros_like(original_rates)

            break

    # average_rate_per_flow = np.mean(all_iterations_rates, axis=0)
    # average_capacity_per_flow = np.mean(all_iterations_capacities, axis=0)
    # deltas = np.array([all_iterations_rates[i]-all_iterations_rates[i-1] for i in range(1, len(all_iterations_rates[0]))])
    # delta_average = np.mean(deltas, axis=1)
    all_rates_av = []
    for i in range(1, all_iterations_rates.shape[0]):
        row = all_iterations_rates[i, :]
        if np.any(row != 0):
            rates_av = np.mean(row[row != 0])
            all_rates_av.append(rates_av)

    return all_possible_paths, deltas_average, all_iterations_rates, all_rates_av, all_iterations_capacities_same_route


def calc_rate_with_path(env, paths):
    """
    Calculates rate for every flow based on the path chosen for the flow,other flows
    Interference is calculated by 1/(r**2), where r is the distance between two *links*
            Capacity of each link is effected:  capacity = bandwidth*log2(1+SINR) assuming unit transmission power

    :param env: env with all the information based on interference
    :param paths: paths chosen for all flows
    """
    all_link_interference = copy.deepcopy(env.cumulative_link_interference)  # cumulative link interference after path alocation for every link
    all_link_capacity = np.zeros_like(all_link_interference)
    flows_bottleneck = np.zeros(len(env.flows))

    # Count number of appearances for every link
    # We need to know how many transmissions appear on one link in order to divide capacity
    link_counter = Counter()
    for path_idx, path in enumerate(paths):
        # Iterate over consecutive nodes in the path and update the counter
        for i in range(len(path) - 1):
            link = (path[i], path[i + 1])
            link_counter[link] += 1

    active_links = list(link_counter.keys())
    active_links_indices = np.array([env.eids[link] for link in active_links])

    for flow_idx, path in enumerate(paths):
        path_capacities = np.zeros(len(path) - 1)
        for i in range(len(path) - 1):
            s, d = path[i], path[i+1]
            link_interference = np.sum((env.interference_map[:, env.eids[s, d]])[active_links_indices])  #
            # I_l calculates interference from all other links, Only those who make a transmission

            trx_power = env.trx_power[env.eids[s, d]]  # P_l
            # link_interference = all_link_interference[env.eids[s, d]]  # cumulative interference for link l
            sinr = trx_power / (link_interference + 1)  # SINR_l
            link_capacity = np.minimum(env.bandwidth_edge_list[env.eids[s, d]],
                                       np.maximum(1, np.floor(env.bandwidth_edge_list[env.eids[s, d]] * np.log2(1 + sinr))))
            # If few flows share the same link divide link capacity
            path_capacities[i] = link_capacity // link_counter[(s, d)]

        # Calc Bottleneck
        idx = np.argmin(path_capacities)
        min_val = path_capacities[idx]
        flows_bottleneck[flow_idx] = min_val

    flows_rates = [min(packet, flow_bottleneck) for packet, flow_bottleneck in zip(env.packets, flows_bottleneck)]

    return flows_rates


def create_test_net():

    Gbase = nx.Graph()
    Gbase.add_nodes_from([0, 1, 2])
    Gbase.add_edges_from(
        [(0, 1), (0, 2), (1, 2)])
    A = np.array(nx.to_numpy_matrix(Gbase))
    A = np.clip(A + A.T, a_min=0, a_max=1)
    G = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
    pos = nx.spring_layout(G, seed=1234)
    pos = np.stack(list(pos.values()), axis=0)

    return G, A, pos


def paths_bottleneck(paths, G, interference_map, bandwidth_edge_list, eids, links_transmission_power):
    # Count number of appearances for every link
    # We need to know how many transmissions appear on one link in order to divide capacity
    link_counter = Counter()
    for path in paths:
        # Iterate over consecutive nodes in the path and update the counter
        for i in range(len(path) - 1):
            link = (path[i], path[i + 1])
            link_counter[link] += 1
    # ---------------------------- #

    paths_bottleneck = []

    # G, interference_map, bandwidth_edge_list, eids, links_transmission_power = create_test_graph()

    for path in paths:
        path_capacities = []
        for i in range(len(path) - 1):
            s, d = path[i], path[i+1]
            link_interference = np.sum(interference_map[:, eids[s, d]])  # I_l calculates interference from all other links
            transmission_power = links_transmission_power[eids[s, d]]  # P_l transmission Power from s to d
            sinr = transmission_power / (link_interference + 1)  # SINR_l assuming noise with unit variance
            link_bandwidth = bandwidth_edge_list[eids[s, d]]  # Bandwidth of the link
            cap = link_bandwidth * np.log2(1 + sinr)

            # Shanon
            link_capacity = np.minimum(link_bandwidth,
                                       np.maximum(1, np.floor(link_bandwidth * np.log2(1 + sinr))))

            # If multiple links transmit on same link, divide the capacity
            # TODO: with current approch the capacity is not fully used
            link_capacity = link_capacity // link_counter[(s, d)]

            path_capacities.append(link_capacity)

        path_bottleneck = np.min(path_capacities)
        paths_bottleneck.append(path_bottleneck)

    return G, paths_bottleneck


def show_graph2(G, pos, paths, rates, transmission_power, show_fig=True):
    """Draw the global graph with edge weights."""
    plt.figure(figsize=(12, 4))

    plt.subplot(2, 2, 1)
    plt.title("Test Communication net flow rates")
    plt.grid(True)
    # Draw nodes
    node_labels = {paths[0][0]: "s1", paths[0][-1]: "d1", paths[1][0]: "s2", paths[1][-1]: "d2"}
    nx.draw_networkx_nodes(G, pos, node_color="tab:blue")
    nx.draw_networkx_labels(G, pos, labels=node_labels)
    # Draw edges
    nx.draw_networkx_edges(G, pos)
    colors = ["red", "green"]
    for j, path in enumerate(paths):
        links = []
        for i in range(len(path) - 1):
            links.append((path[i], path[i+1]))
        nx.draw_networkx_edges(G, pos, edgelist=links, edge_color=colors[j], width=2, style='dashed')
    # Draw edge labels (weights)
    assign_rate_attribute(G, paths, rates, transmission_power)
    edge_labels = nx.get_edge_attributes(G, 'rate')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.axis('off')

    # ----------------------------------------------#

    # Second subplot: Same graph with 'bandwidth' labels

    plt.subplot(2, 2, 2)
    plt.title("Test Communication net transmission power")
    # Assign 'bandwidth' attributes
    nx.draw_networkx_nodes(G, pos, node_color="tab:blue")
    nx.draw_networkx_labels(G, pos, labels=node_labels)
    nx.draw_networkx_edges(G, pos)
    for j, path in enumerate(paths):
        links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        nx.draw_networkx_edges(G, pos, edgelist=links, edge_color=colors[j], width=2, style='dashed')
    edge_labels = nx.get_edge_attributes(G, 'transmission_power')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.axis('off')

    plt.subplot(2, 2, 3)
    nx.draw_networkx(G, pos, node_color="tab:blue")

    plt.tight_layout()

    if show_fig:
        plt.show()


def assign_rate_attribute(graph, paths, rates, transmission_power):
    # Iterate over paths and rates
    for path, rate in zip(paths, rates):
        # Iterate over consecutive nodes in the path
        for u, v in zip(path[:-1], path[1:]):
            # Check if the directed edge (u, v) is in the graph
            if graph.has_edge(u, v):
                # Assign the rate as an attribute to the edge
                graph[u][v]['rate'] = rate
    i = 0
    for edge in graph.edges():
        if edge[0] < edge[1]:
            graph.edges[edge]['transmission_power'] = transmission_power[i].round(2)
            i += 1


def plot_all_iteration_rates(flows_rate_av):
    plt.figure()
    # deltas_average = []
    # flows_rate = np.ndarray(copy.deepcopy(flows_rate_av))
    for i in range(len(flows_rate_av)):
        row = flows_rate_av[i]
        plt.scatter(np.arange(1, len(row)+1), row, label=f'{(i+1)*10} Flows')
        # row = all_iteration_rates[i, :]
        # if np.any(row != 0):
        #     rates_av = np.mean(row[row != 0])
        #     all_rates_av.append(rates_av)
            # nonzero_indices = np.nonzero(row)
            # deltas = all_iteration_rates[i][nonzero_indices] - all_iteration_rates[i-1][nonzero_indices]
            # deltas_average.append(np.mean(deltas))

    # plt.scatter(np.arange(1, len(all_rates_av) + 1), all_rates_av)
    plt.title(f"Average Flow Rate")
    plt.xlabel("Iteration Number")
    plt.ylabel("Rate[Mbps]")
    plt.legend()
    plt.grid(True)
    plt.show()


def filter_colors(color_names):
    filtered_colors = []
    for color_name in color_names:
        rgb_string = mcolors.CSS4_COLORS[color_name]
        # Convert RGB string to integers
        r = int(rgb_string.lstrip('#')[0:2], 16)
        g = int(rgb_string.lstrip('#')[2:4], 16)
        b = int(rgb_string.lstrip('#')[4:6], 16)
        # Convert color to grayscale to check brightness
        brightness = (r + g + b) / 3  # Using average of RGB values as brightness
        # You can adjust the threshold according to your needs
        if brightness < 130:  # You can adjust the threshold according to your needs
            filtered_colors.append(color_name)
    return filtered_colors


def plot_rates_delays(all_av_raz, all_av_aviv, number_of_flows, type="rate"):
    random.seed()
    color_list = list(mcolors.CSS4_COLORS.keys())
    color = random.choice(color_list)

    plt.figure()
    plt.plot(np.arange(1, len(all_av_raz) + 1), all_av_raz, color=color, linestyle='--',
             label='Original Route')
    plt.plot(np.arange(1, len(all_av_aviv) + 1), all_av_aviv, color=color, label='Demand Slicing')

    plt.grid(True)
    plt.legend()
    plt.xlabel("Iteration Number")
    if type == "rate":
        plt.ylabel("Rate[Mbps]")
        plt.title(f"Average Rate {number_of_flows} Flows")
    if type == "delay":
        plt.ylabel("Delay[timesteps]")
        plt.title(f"Average Delay {number_of_flows} Flows")
    plt.show()

    """
    plt.figure()
    for flow in range(number_of_flows):
        column = all_iteration_rates[:, flow]
        column = column[column != 0]
        plt.plot(np.arange(1, len(column) + 1), column, label=f"Flow:{flow+1}")
        plt.plot(np.arange(1, len(column) + 1), rates_for_full_demand[flow] * np.ones_like(column),
                 linestyle='--', label='Full Demand')
    plt.legend()
    plt.xlabel("Iteration Number")
    plt.ylabel("Rate[Mbps]")
    plt.title("Flow Rate")
    plt.show()

    """


def plot_all_rates_delays(flows_average_aviv, flows_average_raz, flows_number, type="rate"):
    random.seed()

    # Generate color list
    color_list = list(mcolors.CSS4_COLORS.keys())

    # Filter colors based on brightness
    filtered_color_list = filter_colors(color_list)

    plt.figure()
    used_colors = []
    for idx, (rate_av, full_demand_rate) in enumerate(zip(flows_average_aviv, flows_average_raz)):
        chose_unused = False
        while not chose_unused:
            color = random.choice(filtered_color_list)
            if color not in used_colors:
                chose_unused = True
                used_colors.append(color)

        plt.plot(np.arange(1, len(rate_av) + 1), rate_av, color=color,
                 label=f'Demand slicing {flows_number[idx]} Flows')
        plt.plot(np.arange(1, len(full_demand_rate) + 1), full_demand_rate, color=color, linestyle='--',
                label=f'Original Route{flows_number[idx]} Flows')

    plt.grid(True)
    plt.legend()
    plt.xlabel("Iteration Number")
    if type == "rate":
        plt.ylabel("Rate[Mbps]")
        plt.title(f"Average Rate")
    if type == "delay":
        plt.ylabel("Delay[timesteps]")
        plt.title(f"Average Delay")
    plt.show()


def plot_flow_rate(all_iterations_capacities_same_route):
    plt.figure()
    for flow_idx in range(all_iterations_capacities_same_route.shape[1]):
        column = all_iterations_capacities_same_route[:, flow_idx]
        column = column[column != 0]
        plt.scatter(np.arange(1, len(column) + 1), column, label=f'Flow:{flow_idx+1}')
    plt.xlabel("Iteration number")
    plt.ylabel("Rate[Mbps]")
    plt.title(f"Flow Rates Original Route, {all_iterations_capacities_same_route.shape[1]} flows ")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_first_changes(before_changes, first_idx_change, first_changes_aviv, first_changes_raz, flows_number, type="Rate"):

    seed = 1
    # Generate color list
    color_list = list(mcolors.CSS4_COLORS.keys())

    # Filter colors based on brightness
    filtered_color_list = filter_colors(color_list)

    plt.figure()
    aviv_changes_average = np.mean(first_changes_aviv, axis=0)
    raz_changes_average = np.mean(first_changes_raz, axis=0)
    first_idx_average = np.mean(first_idx_change, axis=0)
    before_changes_average = np.mean(before_changes, axis=0)

    for k, flows_num in enumerate(flows_number):
        random.seed(seed+k)
        color = random.choice(filtered_color_list)

        view_step_aviv = np.repeat(before_changes_average[k], first_idx_average[k] - 1)
        # view_step_raz = np.repeat(before_changes_average[k], first_idx_average[k] - 1)

        plt.plot(np.arange(1, int(first_idx_average[k])), view_step_aviv, color=color, linestyle='--')
        plt.scatter(int(first_idx_average[k]), aviv_changes_average[k], marker="o", label=f'Aviv {flows_num} Flows',
                    color=color)

        # plt.plot(np.arange(1, int(first_idx_average[k]) + 1), view_step_raz)
        plt.scatter(int(first_idx_average[k]), raz_changes_average[k], marker="*", label=f'Raz {flows_num} Flows',
                    color=color)

    plt.grid(True)
    plt.legend(loc='lower left')
    plt.title(f"Average {type} First Change")
    plt.xlabel("Average Iteration Number")
    plt.ylabel(f"Average {type}")
    # plt.show()

    # Save image
    plt.savefig(f'First_Jump_{type}_plot.png')

    # Close the plot
    plt.close()


def plot_half_finished_changes(before_changes, all_half_flows_rates_index_aviv , all_half_flows_rates_index_raz,
                               all_half_flows_rates_aviv, all_half_flows_rates_raz, flows_number, type="Rate"):
    seed = 1
    # Generate color list
    color_list = list(mcolors.CSS4_COLORS.keys())

    # Filter colors based on brightness
    filtered_color_list = filter_colors(color_list)

    plt.figure()
    aviv_changes_average = np.mean(all_half_flows_rates_aviv, axis=0)
    raz_changes_average = np.mean(all_half_flows_rates_raz, axis=0)
    half_idx_average_aviv = np.mean(all_half_flows_rates_index_aviv, axis=0)
    half_idx_average_raz = np.mean(all_half_flows_rates_index_raz, axis=0)
    before_changes_average = np.mean(before_changes, axis=0)

    for k, flows_num in enumerate(flows_number):
        random.seed(seed+k)
        color = random.choice(filtered_color_list)
        max_index = int(max(half_idx_average_aviv[k], half_idx_average_raz[k]))

        view_step_aviv = np.repeat(before_changes_average[k], max_index - 1)

        plt.plot(np.arange(1, max_index), view_step_aviv, color=color, linestyle='--')
        plt.scatter(max_index, aviv_changes_average[k], marker="o", label=f'Aviv {flows_num} Flows',
                    color=color)

        plt.scatter(max_index, raz_changes_average[k], marker="*", label=f'Raz {flows_num} Flows',
                    color=color)

    plt.grid(True)
    plt.legend(loc='lower left')
    plt.title(f"Average {type} With Half Flows 'Alive'")
    plt.xlabel("Average Iteration Number")
    plt.ylabel(f"Average {type}")
    # plt.show()

    # Save image
    plt.savefig(f'{type}_when_half_finished_plot.png')

    # Close the plot
    plt.close()


if __name__ == "__main__":

    # create_subplot()
    MODEL_PATH = os.path.join("DIAMOND", "pretrained", "model_20221113_212726_480.pt")

    # params
    num_nodes = 60  # 5
    num_edges = 70  # 10
    num_actions = 4
    temperature = 1.2
    num_episodes = 10
    episode_from = 7500
    nb3r_steps = 100
    min_flow_demand = 1500  # 100 10,000
    max_flow_demand = 1500  # 800 20,000
    min_capacity = 400
    max_capacity = 400
    og_seed = 26

    trx_power_mode = 'equal'
    rayleigh_scale = 1
    max_trx_power = 10
    channel_gain = 1

    packets_per_iteration = 1e3
    number_time_slices = 7

    GRAPH_MODE = 'random'

    alg = DIAMOND(grrl_model_path=MODEL_PATH, nb3r_tmpr=temperature, nb3r_steps=nb3r_steps)

    flows_number = [20, 30, 40, 50]

    number_of_runs = 20

    all_first_difference_rates = np.zeros((number_of_runs, len(flows_number)))
    all_first_difference_delays = np.zeros((number_of_runs, len(flows_number)))

    first_rate_idx_changes = np.zeros((number_of_runs, len(flows_number)))
    first_delay_idx_changes = np.zeros((number_of_runs, len(flows_number)))
    first_rate_changes_aviv = np.zeros((number_of_runs, len(flows_number)))
    first_rate_changes_raz = np.zeros((number_of_runs, len(flows_number)))
    first_delays_changes_aviv = np.zeros((number_of_runs, len(flows_number)))
    first_delays_changes_raz = np.zeros((number_of_runs, len(flows_number)))
    before_changes_rate = np.zeros((number_of_runs, len(flows_number)))
    before_changes_delay = np.zeros((number_of_runs, len(flows_number)))

    # Rate
    all_half_flows_rates_aviv = np.zeros((number_of_runs, len(flows_number)))
    all_half_flows_rates_raz = np.zeros((number_of_runs, len(flows_number)))
    all_half_flows_rates_index_aviv = np.zeros((number_of_runs, len(flows_number)))
    all_half_flows_rates_index_raz = np.zeros((number_of_runs, len(flows_number)))

    # Delay
    all_half_flows_delays_aviv = np.zeros((number_of_runs, len(flows_number)))
    all_half_flows_delays_raz = np.zeros((number_of_runs, len(flows_number)))

    # Start Simulation loop
    for run_number in range(number_of_runs):
        seed = og_seed + run_number * 10

        # flows_rate_average_aviv = []
        # flows_rate_average_raz = []
        # flows_delay_average_aviv = []
        # flows_delay_average_raz = []
        # first_changes_rate = []
        # first_changes_delay = []

        for i, num_flows in enumerate(flows_number):
            # print(f'Graph Mode:{GRAPH_MODE}, trx_power_mode:{trx_power_mode}, flows:{num_flows}\n\n')

            env = generate_env(num_nodes=num_nodes, num_edges=num_edges,
                               num_flows=num_flows, min_flow_demand=min_flow_demand,
                               max_flow_demand=max_flow_demand, min_capacity=min_capacity,
                               max_capacity=max_capacity, seed=seed, graph_mode=GRAPH_MODE,
                               trx_power_mode=trx_power_mode, rayleigh_scale=rayleigh_scale,
                               max_trx_power=max_trx_power, channel_gain=channel_gain)
            # env.show_graph()

            start_time = time.time()
            (all_rates_av_aviv, all_delays_av_aviv, all_rates_av_raz, all_delays_av_raz, half_flows_rates_index_aviv,
             half_flows_rates_index_raz) = transmission_slicing(env=env, alg=alg,
                                                        packets_per_iteration=packets_per_iteration,
                                                        min_flow_demand=min_flow_demand,
                                                        max_flow_demand=max_flow_demand,
                                                        min_capacity=min_capacity,
                                                        max_capacity=max_capacity, seed=seed,
                                                        graph_mode=GRAPH_MODE,
                                                        trx_power_mode=trx_power_mode,
                                                        rayleigh_scale=rayleigh_scale,
                                                        max_trx_power=max_trx_power,
                                                        channel_gain=channel_gain)
            # Saving arrays
            # np.save(f'all_rates_av_G_{GRAPH_MODE}_N_{num_nodes}_E_{num_edges}_F_{num_flows}_mincap_{min_capacity}_maxcap_{max_capacity}_mindemand_{min_flow_demand}_max_demand_{max_flow_demand}_packs_iter_{packets_per_iteration}.npy', all_rates_av_aviv)
            # np.save(f'all_delays_av_G_{GRAPH_MODE}_N_{num_nodes}_E_{num_edges}_F_{num_flows}_mincap_{min_capacity}_maxcap_{max_capacity}_mindemand_{min_flow_demand}_max_demand_{max_flow_demand}_packs_iter_{packets_per_iteration}.npy', all_delays_av_aviv)

            # Plotting rates delays
            # plot_rates_delays(all_rates_av_raz, all_rates_av_aviv, num_flows, type="rate")
            # plot_rates_delays(all_delays_av_raz, all_delays_av_aviv, num_flows, type="delay")

            # flows_rate_average_aviv.append(all_rates_av_aviv)
            # flows_rate_average_raz.append(all_rates_av_raz)
            #
            # flows_delay_average_aviv.append(all_delays_av_aviv)
            # flows_delay_average_raz.append(all_delays_av_raz)

            (first_change_index_rate, first_change_index_delay, first_change_rate, first_change_delay,
             before_change_rates, before_change_delays) = first_change(all_rates_av_aviv, all_rates_av_raz,
                                                                       all_delays_av_aviv, all_delays_av_raz)

            """
            Add elements to global arrays
            """

            """
            First changes
            """
            first_rate_idx_changes[run_number, i] = first_change_index_rate
            first_delay_idx_changes[run_number, i] = first_change_index_delay
            first_rate_changes_aviv[run_number, i], first_rate_changes_raz[run_number, i] = first_change_rate[0], first_change_rate[1]
            first_delays_changes_aviv[run_number, i], first_delays_changes_raz[run_number, i] = first_change_delay[0], first_change_delay[1]
            before_changes_rate[run_number, i] = before_change_rates
            before_changes_delay[run_number, i] = before_change_delays

            """
            Changes when Half finished
            """

            # rate
            all_half_flows_rates_index_aviv[run_number, i], all_half_flows_rates_index_raz[run_number, i] = half_flows_rates_index_aviv, half_flows_rates_index_raz
            all_half_flows_rates_aviv[run_number, i], all_half_flows_rates_raz[run_number, i] = all_rates_av_aviv[half_flows_rates_index_aviv], all_rates_av_raz[half_flows_rates_index_raz]

            # delay
            all_half_flows_delays_aviv[run_number, i], all_half_flows_delays_raz[run_number, i] = all_delays_av_aviv[
                half_flows_rates_index_aviv], all_delays_av_raz[half_flows_rates_index_raz]


            run = time.time() - start_time
            print("took {} s  with {} flows, Graph mode : {}, itr: {}\n".format(time.strftime('%H:%M:%S', time.gmtime(run)), num_flows, GRAPH_MODE, run_number))

        """
        Logging Results
        """
        # log_data_points(d1=first_rate_changes_aviv[run_number, 0], d2=first_rate_changes_raz[run_number, 0],
        #                 d3=first_rate_changes_aviv[run_number, 1], d4=first_rate_changes_raz[run_number, 1],
        #                 d5=first_rate_changes_aviv[run_number, 2], d6=first_rate_changes_raz[run_number, 2],
        #                 d7=first_delays_changes_aviv[run_number, 0], d8=first_delays_changes_raz[run_number, 0],
        #                 d9=first_delays_changes_aviv[run_number, 1], d10=first_delays_changes_raz[run_number, 1],
        #                 d11=first_delays_changes_aviv[run_number, 2], d12=first_delays_changes_raz[run_number, 2],
        #                 iteration=run_number, flows_number=flows_number)

        # Saving arrays
        # np.save(f'flows_rate_all_flows_numbers_G_{GRAPH_MODE}_N_{num_nodes}_E_{num_edges}_F_{flows_number}_mincap_{min_capacity}_maxcap_{max_capacity}_mindemand_{min_flow_demand}_max_demand_{max_flow_demand}_packs_iter_{packets_per_iteration}.npy', flows_rate_average_aviv)
        # np.save(f'flows_delay_all_flows_numbers_G_{GRAPH_MODE}_N_{num_nodes}_E_{num_edges}_F_{flows_number}_mincap_{min_capacity}_maxcap_{max_capacity}_mindemand_{min_flow_demand}_max_demand_{max_flow_demand}_packs_iter_{packets_per_iteration}.npy', flows_delay_average_aviv)

        # plot_all_iteration_rates(flows_rate_average)
        #plot_all_rates_delays(flows_average_aviv=flows_rate_average_aviv, flows_average_raz=flows_rate_average_raz,
                             # flows_number=flows_number, type="rate")
        #plot_all_rates_delays(flows_average_aviv=flows_delay_average_aviv, flows_average_raz=flows_delay_average_raz,
                             # flows_number=flows_number, type="delay")

        # all_first_difference_rates[run_number] = np.array(first_changes_rate)
        # all_first_difference_delays[run_number] = np.array(first_changes_delay)

        print(f"Finished run {run_number}\n")

    # print("finished")

    # np.save(f'rates_first_difference_{num_nodes}N_{num_edges}E.npy', all_first_difference_rates)
    # np.save(f'delays_first_difference_{num_nodes}N_{num_edges}E.npy', all_first_difference_delays)

    # First Jumps

    plot_first_changes(before_changes=before_changes_rate, first_idx_change=first_rate_idx_changes,
                       first_changes_aviv=first_rate_changes_aviv, first_changes_raz=first_rate_changes_raz,
                       flows_number=flows_number, type="Rate")

    plot_first_changes(before_changes=before_changes_delay, first_idx_change=first_delay_idx_changes,
                       first_changes_aviv=first_delays_changes_aviv, first_changes_raz=first_delays_changes_raz,
                       flows_number=flows_number, type="Delay")

    # Half finished

    plot_half_finished_changes(before_changes=before_changes_rate,
                               all_half_flows_rates_index_aviv=all_half_flows_rates_index_aviv,
                               all_half_flows_rates_index_raz=all_half_flows_rates_index_raz,
                               all_half_flows_rates_aviv=all_half_flows_rates_aviv,
                               all_half_flows_rates_raz=all_half_flows_rates_raz,
                               flows_number=flows_number, type="Rate")

    # for delays indices are the same as rates, when half finished

    plot_half_finished_changes(before_changes=before_changes_delay,
                               all_half_flows_rates_index_aviv=all_half_flows_rates_index_aviv,
                               all_half_flows_rates_index_raz=all_half_flows_rates_index_raz,
                               all_half_flows_rates_aviv=all_half_flows_delays_aviv,
                               all_half_flows_rates_raz=all_half_flows_delays_raz,
                               flows_number=flows_number, type="Delay")

    # Close the SummaryWriters when finished with all iterations
    # writer_Rates.close()
    # writer_Delays.close()

    print("Done")


















