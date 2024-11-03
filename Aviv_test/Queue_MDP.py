import numpy as np
import matplotlib.pyplot as plt
from environment import generate_env
from diamond import DIAMOND
import os
import random
import copy
from collections import Counter
# from arrivals import TCPRenoSimulator, poisson_arrivals
from Aviv_test.Traffic_prediction_model import NTP


class NetworkMDP:
    def __init__(self, alg, threshold, arrival_matrix, packets_per_iteration,
                 num_nodes=3,
                 num_edges=3,
                 num_actions=4,
                 num_flows=10,
                 min_flow_demand=100,
                 max_flow_demand=200,
                 min_capacity=10,
                 max_capacity=20,
                 seed=320,
                 graph_mode='abilene',
                 trx_power_mode='equal',
                 rayleigh_scale=1,
                 max_trx_power=10,
                 channel_gain=1,
                 new_flows=None):

        self.alg = alg

        """
        Env params
        """
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.num_actions = num_actions
        self.num_flows = num_flows
        self.min_flow_demand = min_flow_demand
        self.max_flow_demand = max_flow_demand
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.seed = seed
        self.graph_mode = graph_mode
        self.trx_power_mode = trx_power_mode
        self.rayleigh_scale = rayleigh_scale
        self.max_trx_power = max_trx_power
        self.channel_gain = channel_gain

        """
        lists
        """

        """
        initializing params
        """
        # Create first env
        self.env = None
        self.update_env(new_flows=new_flows)
        self.current_flows = copy.deepcopy(self.env.flows)
        self.alive_flows_indices = np.array(list(range(self.num_flows)))
        self.dead_flows_indices = np.array([])

        self.arrival_matrix = np.array(arrival_matrix)
        self.packets_per_iteration = packets_per_iteration
        self.threshold = threshold
        self.buffers = np.array(self.arrival_matrix[0, :])  # assume at t=0 buffer is empty
        self.buffers = np.minimum(np.maximum(self.buffers, 1), self.max_flow_demand)
        self.status = np.ones(self.env.num_flows)  # Initialize all flows to alive (1)
        self.alive_counts = np.zeros(self.env.num_flows)  # Count of how often each flow is alive
        self.total_steps = 0  # Track the total number of steps

        # Update env for demand as first arrival
        self.update_flows()
        self.update_env(new_flows=self.current_flows)

    def update_env(self, new_flows=None):

        env = generate_env(num_nodes=self.num_nodes,
                           num_edges=self.num_edges,
                           num_actions=self.num_actions,
                           num_flows=self.num_flows,
                           min_flow_demand=self.min_flow_demand,
                           max_flow_demand=self.max_flow_demand,
                           min_capacity=self.min_capacity,
                           max_capacity=self.max_capacity,
                           seed=self.seed,
                           graph_mode=self.graph_mode,
                           trx_power_mode=self.trx_power_mode,
                           rayleigh_scale=self.rayleigh_scale,
                           max_trx_power=self.max_trx_power,
                           channel_gain=self.channel_gain,
                           given_flows=new_flows
                           )
        self.env = env

    def update_flows(self):
        # self.env.show_graph()
        new_flows = []
        for flow_id, flow in enumerate(self.current_flows):
            # if self.status[flow_id]:
            new_flow = copy.deepcopy(flow)  # Don't want the original flow to change
            new_flow['packets'] = int(max(0, self.buffers[flow_id]))  # making sure no negative demand
            new_flows.append(new_flow)

        """
        new_flows = []
        for flow_id, flow in enumerate(self.current_flows):
            new_flow = copy.deepcopy(flow)  # Don't want the original flow to change
            new_flow['packets'] = 1000 + 1000*flow_id  # making sure no negative demand
            new_flows.append(new_flow)
        """
        self.current_flows = new_flows

    def calc_rate_with_path(self, env, paths):

        """
        Calculates rate for every flow based on the path chosen for the flow,other flows
        Interference is calculated by 1/(r**2), where r is the distance between two *links*
                Capacity of each link is effected:  capacity = bandwidth*log2(1+SINR) assuming unit transmission power

        :env: env with all the information based on interference
        :param paths: paths chosen for all flows
        @param env: given environment
        """
        all_link_interference = copy.deepcopy(
            env.cumulative_link_interference)  # cumulative link interference after path alocation for every link
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
                s, d = path[i], path[i + 1]
                link_interference = np.sum((env.interference_map[:, env.eids[s, d]])[active_links_indices])  #
                # I_l calculates interference from all other links, Only those who make a transmission

                trx_power = env.trx_power[env.eids[s, d]]  # P_l
                # link_interference = all_link_interference[env.eids[s, d]]  # cumulative interference for link l
                sinr = trx_power / (link_interference + 1)  # SINR_l
                link_capacity = np.minimum(env.bandwidth_edge_list[env.eids[s, d]],
                                           np.maximum(1, np.floor(
                                               env.bandwidth_edge_list[env.eids[s, d]] * np.log2(1 + sinr))))
                # If few flows share the same link divide link capacity
                path_capacities[i] = link_capacity // link_counter[(s, d)]

            # Calc Bottleneck
            idx = np.argmin(path_capacities)
            min_val = path_capacities[idx]
            flows_bottleneck[flow_idx] = min_val

        flows_rates = [min(packet, flow_bottleneck) for packet, flow_bottleneck in zip(env.packets, flows_bottleneck)]

        return np.array(flows_rates)

    def diamond_step(self):
        """
        For "dead" flows we specify a path, still need to go in env for calculations
        """

        if len(self.dead_flows_indices) != 0:
            paths = self.alg(self.env, use_nb3r=False, dead_flow_indices=self.dead_flows_indices)

        else:
            paths = self.alg(self.env, use_nb3r=False)

        # rates = self.env.flows_rate
        delays = self.env.flows_delay

        rates = self.calc_rate_with_path(env=self.env, paths=paths)

        # transmission slicing
        # rates_with_demand_slicing, delays_with_demand_slicing, paths_with_demand_slicing = self.transmission_slicing(
        #                                                                 env=self.env,
        #                                                                 alg=self.alg,
        #                                                                 packets_per_iteration=self.packets_per_iteration
        #                                                                 ,)

        return paths, rates, delays  # , rates_with_demand_slicing, delays_with_demand_slicing

    def update_buffers(self, arrivals):

        paths, rates, delays = self.diamond_step()
        # paths, rates, delays, rates_with_demand_slicing, delays_with_demand_slicing = self.diamond_step()

        # self.buffers[self.alive_flows_indices] += -rates
        self.buffers += arrivals - rates
        self.buffers = np.minimum(np.maximum(self.buffers, 1), self.max_flow_demand)  # no negative buffer

        # Update status based on buffer size
        self.status = np.where(self.buffers >= self.threshold, 1, 0)
        self.alive_flows_indices = np.nonzero(self.buffers >= self.threshold)[0]
        self.dead_flows_indices = np.nonzero(self.buffers < self.threshold)[0]
        self.alive_counts += self.status
        self.total_steps += 1

        return paths, rates, delays  # , rates_with_demand_slicing, delays_with_demand_slicing

    def transmission_slicing(self, env, alg, packets_per_iteration, return_paths=False, num_time_steps=60):
        """
        :param env: env after setting real demands
        :param alg: DIAMOND alg
        :param packets_per_iteration: no. of packets we stream every iter for comparison
        :para return_paths: return the paths in each iteration
        :@return: average rate/delay for every flow for all iterations, paths for first iteration
        """
        counter = env.num_flows  # For Debugging

        # current demands are different because rates are different. demand = demand - rate_for_iteration
        current_demand = copy.copy(env.packets)

        # Aviv
        all_iteration_rates = [[] for _ in range(env.num_flows)]  # np.zeros((k, env.num_flows))
        all_iteration_delay = [[] for _ in range(env.num_flows)]  # np.zeros((k, env.num_flows))
        all_iterations_average_rate = []
        all_iterations_average_delay = []

        average_rates = np.zeros(env.num_flows)
        average_delays = np.zeros(env.num_flows)

        all_possible_paths = [[] for _ in range(env.num_flows)]

        # Initializing "new flows" dict to work with every iteration, different flows are "alive" for aviv\raz
        new_flows = copy.deepcopy(env.flows)


        # Mapping flows to indices

        remaining_flows_indices = np.array(list(range(env.num_flows)))


        # inserting demand for new flows, the demand will stay for all iters
        for flow in new_flows:
            flow["packets"] = packets_per_iteration


        # Starting main loop

        finished = False

        i = 0  # Counter for placements in global lists
        # for i in range(k):
        while not finished:

            # If we run more than asked stop
            if i >= num_time_steps:
                break

            """
            Aviv
            """
            if not finished:

                # Setting env
                env_2 = generate_env(num_nodes=self.num_nodes,
                                        num_edges=self.num_edges,
                                        num_actions=self.num_actions,
                                        num_flows=self.num_flows,
                                        min_flow_demand=self.min_flow_demand,
                                        max_flow_demand=self.max_flow_demand,
                                        min_capacity=self.min_capacity,
                                        max_capacity=self.max_capacity,
                                        seed=self.seed,
                                        graph_mode=self.graph_mode,
                                        trx_power_mode=self.trx_power_mode,
                                        rayleigh_scale=self.rayleigh_scale,
                                        max_trx_power=self.max_trx_power,
                                        channel_gain=self.channel_gain,
                                        given_flows=new_flows)

                # Aviv
                diamond_paths = alg(env_2, use_nb3r=True)
                flows_rate_per_iter = env_2.flows_rate
                flows_rate_per_iter_test = self.calc_rate_with_path(env=env_2, paths=diamond_paths)

                flows_delay_per_iter = env_2.flows_delay

                # For debugging
                if len(remaining_flows_indices) != counter:
                    # few can finish in same iteration so not good for tracking
                    counter -= 1
                    # print("one more fell aviv")

                """
                Updating Rates, Delay arrays
                """

                # rates is a list of remaining flows rate given in  the last iteration
                if i == 0:
                    # Aviv
                    rates = flows_rate_per_iter
                    delays = flows_delay_per_iter

                # Aviv
                rates[np.array(remaining_flows_indices)] = flows_rate_per_iter
                delays[np.array(remaining_flows_indices)] = flows_delay_per_iter

                # Adding average to the list
                all_iterations_average_delay.append(np.mean(delays[delays != 0]))
                all_iterations_average_rate.append(np.mean(rates[rates != 0]))

                # all_iteration_rates[i].append(list(rates))  # [i] = rates
                # all_iteration_delay[i].append(list(delays))  # [i] = delays
                for flow_id in range(env.num_flows):
                    all_iteration_rates[flow_id].append(rates[flow_id])
                    all_iteration_delay[flow_id].append(delays[flow_id])

                """
                Adding current paths to paths list
                """
                # Aviv
                for idx, route in zip(remaining_flows_indices, diamond_paths):
                    all_possible_paths[idx].append(route)

                # For flows that finished we give the same path
                for flow_idx, path_list in enumerate(all_possible_paths):
                    if len(path_list) != i + 1:
                        all_possible_paths[flow_idx].append(all_possible_paths[flow_idx][-1])

                # Updating current demand, every iter we subtract transmitted packets from og demand

                current_demand = np.array(current_demand) - rates

                # checking remaining flows
                remaining_flows_demands = current_demand[np.array(remaining_flows_indices)]

                # TODO: think here. for now "alive" flows have same size of packets every iter.
                # for id, flow in enumerate(new_flows_aviv):
                #     flow["packets"] = min(flow["packets"], remaining_flows_demands[id])

                new_flows = [flow for flow, left_demand in zip(env_2.flows, remaining_flows_demands) if
                                  left_demand > 0]

                """
                for flows that finished transmitting we give a rate delay of 0
                """

                # Aviv
                rates = np.array([rate if current_demand[i] > 0 else 0 for i, rate in enumerate(rates)])
                delays = np.array(
                    [delay if current_demand[i] > 0 else 0 for i, delay in enumerate(delays)])

                # Updating remaining flows indices
                remaining_flows_indices = np.nonzero(current_demand > 0)[0]
                num_fallen_flows = env.num_flows - len(remaining_flows_indices)

                # Checking if there are Flows left
                if len(remaining_flows_indices) == 0:
                    finished = True

            i += 1

        """
        for every flow calculate average value for rate/delay for all iterations
        """
        for flow_id in range(env.num_flows):
            # Rates
            flow_rates = np.array(all_iteration_rates[flow_id])
            flow_average_rate = np.mean(flow_rates[flow_rates != 0]).round(3)
            average_rates[flow_id] = flow_average_rate

            # Delays
            flow_delays = np.array(all_iteration_delay[flow_id])
            flow_average_delay = np.mean(flow_delays[flow_delays != 0]).round(3)
            average_delays[flow_id] = flow_average_delay

        if return_paths:
            return all_possible_paths
        else:
            return np.array(average_rates), np.array(average_delays), [all_paths[0] for all_paths in all_possible_paths]

    def step(self, arrivals, seed=10):

        paths, rates, delays = self.update_buffers(arrivals=arrivals)
        # paths, rates, delays, rates_with_demand_slicing, delays_with_demand_slicing = self.update_buffers(arrivals=arrivals)

        self.update_flows()

        self.update_env(new_flows=self.current_flows)

        return paths, rates, delays  # , rates_with_demand_slicing, delays_with_demand_slicing

    def run_episode(self, time_steps):

        # self.env.show_graph()

        # TODO:maybe better to have arrival_matrix as an input to the func instead of an instance to the class

        arrival_matrix = self.arrival_matrix

        # Paths
        all_iterations_paths = [[] for _ in range(self.num_flows)]

        # Rates
        all_iterations_rates = np.zeros((time_steps - 1, self.num_flows))  # time_steps = len(arrival_matrix) and arrival_matrix[0] is init shape of all rates is one less
        all_iterations_rates_with_time_slicing = np.zeros((time_steps - 1, self.num_flows))

        # Delays
        all_iterations_delays = np.zeros((time_steps - 1, self.num_flows))
        all_iterations_delays_with_time_slicing = np.zeros((time_steps - 1, self.num_flows))

        # all_iterations_test_rates = np.zeros((time_steps, self.num_flows))
        # all_iterations_status = np.zeros((time_steps, self.num_flows))

        for t in range(time_steps - 1):

            # transmission slicing
            # one_step_rates, one_step_delays, one_step_paths = self.transmission_slicing(env=self.env,
            #                                                                             alg=self.alg,
            #                                                                             packets_per_iteration=20)

            paths, rates, delays = self.step(arrivals=arrival_matrix[t + 1])
            # paths, rates, delays, rates_with_demand_slicing, delays_with_demand_slicing = self.step(
            #                                                                             arrivals=arrival_matrix[t + 1])   # arrival_matrix[0] is initialization so first new arrivals is [1]

            # Adding paths
            for idx, route in enumerate(paths):
                all_iterations_paths[idx].append(route)

            # Add rates/delays with/without demand slicing
            all_iterations_rates[t, :] = rates
            # all_iterations_rates_with_time_slicing[t, :] = rates_with_demand_slicing
            all_iterations_delays[t, :] = delays
            # all_iterations_delays_with_time_slicing[t, :] = delays_with_demand_slicing


            # all_iterations_test_rates[t, :] = test_rates

            # Add Status
            # all_iterations_status[t, :] = self.status

        # return all_iterations_paths, all_iterations_rates, all_iterations_rates_with_time_slicing, all_iterations_delays, all_iterations_delays_with_time_slicing
        return all_iterations_paths, all_iterations_rates, all_iterations_delays

    def plot_rates_delays(self, all_iterations_rates, all_iterations_rates_with_time_slicing, all_iterations_delays, all_iterations_delays_with_time_slicing):

        # Rates
        mean_rates_per_time_step = np.mean(all_iterations_rates, axis=1)
        mean_rates_with_time_slicing = np.mean(all_iterations_rates_with_time_slicing, axis=1)

        # Delays
        mean_delays_per_time_step = np.mean(all_iterations_delays, axis=1)
        mean_delays_with_time_slicing = np.mean(all_iterations_delays_with_time_slicing, axis=1)

        # Create a figure and two subplots
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))

        # Plotting Rates
        ax[0].plot(mean_rates_per_time_step, label='Without Demand Slicing', color='blue', marker='o')
        ax[0].plot(mean_rates_with_time_slicing, label='With Demand Slicing', color='orange', marker='*')
        ax[0].set_title('Average Flow Rates over Time')
        ax[0].set_xlabel('Time Step')
        ax[0].set_ylabel('Mean Rate')
        ax[0].legend()
        ax[0].grid(True)

        # Plotting Delays
        ax[1].plot(mean_delays_per_time_step, label='Without Demand Slicing', color='blue', marker='o')
        ax[1].plot(mean_delays_with_time_slicing, label='With Demand Slicing', color='orange', marker='*')
        ax[1].set_title('Average Flow Delays over Time')
        ax[1].set_xlabel('Time Step')
        ax[1].set_ylabel('Mean Delay')
        ax[1].legend()
        ax[1].grid(True)

        # Adjust the layout to prevent overlap
        plt.tight_layout()

        # Show the plots
        plt.show()


if __name__ == "__main__":

    MODEL_PATH = os.path.join("DIAMOND", "pretrained", "model_20221113_212726_480.pt")

    alg = DIAMOND(grrl_model_path=MODEL_PATH,
                  nb3r_tmpr=1.2,
                  nb3r_steps=10
                  )

    # Example usage

    threshold = 1

    # arrival_matrix = poisson_arrivals(seed=1, time_steps=60, arrival_rates=arrival_rates, num_flows=len(arrival_rates))
    # arrival_matrix = NTP(model_type='LSTM', generate_matrix=True, num_flows=10, num_time_steps=60) + 100
    # arrival_matrix = np.ones((2, 10)) * 100
    # arrival_matrix = np.array([[100, 50], [100, 50]])
    # arrival_matrix = np.hstack((np.ones((1, 10)).T * 50, np.ones((1, 10)).T * 100))
    arrival_matrix = np.zeros((3, 10))
    arrival_matrix[0, :], arrival_matrix[1, :], arrival_matrix[2, :] = np.arange(100, 200, 10), np.arange(10, 20), np.arange(10)
    # new_flows = [{"source": 0, "destination": 2, "packets": 100}, {"source": 0, "destination": 2, "packets": 50}]
    arrival_matrix = np.random.randint(low=1, high=100, size=(5, 5))
    mdp = NetworkMDP(alg=alg,
                     threshold=threshold,
                     arrival_matrix=arrival_matrix,
                     packets_per_iteration=20,
                     num_nodes=5,
                     num_edges=10,
                     num_actions=4,
                     num_flows=np.shape(arrival_matrix)[1],
                     min_flow_demand=100,
                     max_flow_demand=200,
                     min_capacity=10,
                     max_capacity=20,
                     seed=123,
                     graph_mode='random',
                     trx_power_mode='equal',
                     rayleigh_scale=1,
                     max_trx_power=10,
                     channel_gain=1,
                     new_flows=None
                     )

    (all_iterations_paths,
     all_iterations_rates,
     all_iterations_delays) = mdp.run_episode(time_steps=len(arrival_matrix))


    print('finished')
