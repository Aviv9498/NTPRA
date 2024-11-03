import networkx as nx
from matplotlib import pyplot as plt

import sys
import numpy as np
from pprint import pprint
from environment.utils import get_k_paths, link_queue_history_using_mac_protocol, init_seed, calc_transmission_rate


class GraphEnvPower:
    """
        TODO: update description

        Class represents the graph environment for Combinatorial Optimization. G=(V,E) directed-weighted graph
        All flow demands are received together and to be allocated at once

        Inputs:
        graph topology
        @param adjacency_matrix: |V|x|V| binary matrix
        @param bandwidth_matrix : |V|x|V| int matrix
        @param interference_matrix: |V|x|V| matrix of edge lists as pairs (u,v)
        @param node_positions: (optional) node's geographic position: |V|x2 np.array tuples (x, y)
        @param flows: list of flow demands {(s_i, d_i, p_i, t_i)}_i=1^F
        @param k: action size - k possible transmission routs

        1. Simulates transmission of each flow on the graph
        2. Returns reward(score) for the choice
        """

    def __init__(self,
                 adjacency_matrix,
                 bandwidth_matrix,
                 interference_matrix,
                 node_positions,
                 flows,
                 k,
                 reward_balance=0.8,
                 seed=37,
                 direction="minimize",
                 normalize_capacity=True,
                 received_interference_map=None,
                 **kwargs):
        super(GraphEnvPower, self).__init__()

        self.trx_power = None
        self.seed = seed
        init_seed(seed)

        # received
        self.adjacency_matrix = adjacency_matrix
        self.bandwidth_matrix = bandwidth_matrix
        self.binary_interference_matrix = interference_matrix if interference_matrix is not None else np.ones(
            adjacency_matrix.shape)
        self.flows = flows
        self.packets = np.array([f["packets"] for f in self.flows])
        self.node_positions = np.array(node_positions)
        self.k = k  # action size (k paths to choose from)
        self.kwargs = kwargs
        self.reward_balance = reward_balance
        self.normalize_capacity = normalize_capacity
        self.received_interference_map = received_interference_map
        self.direction = direction

        # attributes
        self.graph: nx.DiGraph
        self.graph_pos = None  # dict(int: np.array)
        self.nodes = None
        self.num_nodes = None
        self.num_edges = None
        self.num_flows = len(self.flows)

        self.max_capacity = np.max(self.bandwidth_matrix)
        self.max_demand = np.max([f["packets"] for f in self.flows])

        self.possible_actions = [[] for _ in range(len(self.flows))]
        self.history = []
        self.allocated = []  # [False] * len(self.flows)
        self.prev_reward = None
        self.flows_delay = [0 for _ in range(len(self.flows))]
        self.flows_rate = [0 for _ in range(len(self.flows))]

        # data
        self.node_feature_size = 2
        self.edge_feature_size = 3

        self.path_bank = dict()

        # graph data
        self.interference_map = None
        self.current_link_interference = None
        self.links_length = None
        self.cumulative_link_interference = None
        self.current_link_capacity = None
        self.current_link_queue = None
        self.bandwidth_edge_list = None
        self.demands = None
        self.last_action = np.zeros_like(self.adjacency_matrix)

        # for tf env
        self.tf_env = False
        self.num_sps = None
        self.firsts = []
        self.seconds = []
        self.gen_first_second()

        # initialization once
        self.__create_graph()
        self.__calc_possible_actions()
        self.flows_bottleneck = self.__calc_flows_bottleneck()

    def gen_first_second(self):
        for i, u in enumerate(self.adjacency_matrix):
            for j, v in enumerate(u):
                if v != 0:
                    self.firsts.append(i)
                    self.seconds.append(j)

    def gen_edge_data(self):
        self.eids = dict()   # Dictionary to store edge IDs. It uses tuples (u, v) as keys and assigns a unique ID to each edge.
        self.id_to_edge = []  # List to store the mapping between edge IDs and the corresponding edges (tuples)
        self.bandwidth_edge_list = []  # List to store the bandwidth capacity for each edge.
        self.link_pos = []  # List to store the positions of edges (the midpoint between nodes connected by the edge).
        self.links_length = []  # List to store the lengths of edges (Euclidean distance between nodes).
        self.graph_pos = dict()
        spring_pos = nx.spring_layout(self.graph)
        id = 0
        for u in range(self.num_nodes):
            for v in range(u + 1, self.num_nodes):
                if self.adjacency_matrix[u, v]:
                    # edge id
                    self.eids[(u, v)] = id
                    self.eids[(v, u)] = id
                    id += 1

                    # node position
                    if self.node_positions is not None:
                        self.graph_pos[u] = np.array(self.node_positions[u])
                        self.graph_pos[v] = np.array(self.node_positions[v])
                    else:
                        self.graph_pos[u] = np.array(spring_pos[u])
                        self.graph_pos[v] = np.array(spring_pos[v])

                    # edge position
                    self.link_pos.append(np.mean([self.graph_pos[u], self.graph_pos[v]], axis=0))
                    self.id_to_edge.append((u, v))
                    self.links_length.append(np.linalg.norm(self.graph_pos[u] - self.graph_pos[v]))

                    # capacity matrix
                    self.bandwidth_edge_list.append(self.bandwidth_matrix[u, v])

        self.link_pos = np.array(self.link_pos)
        self.links_length = np.array(self.links_length)
        self.bandwidth_edge_list = np.array(self.bandwidth_edge_list)
        self.current_link_capacity = self.bandwidth_edge_list.copy()

    def init_edge_data(self):
        """
        returns interference map
        """
        L = self.num_edges // 2
        self.interference_map = np.zeros((L, L))
        self.current_link_interference = np.zeros(L)
        self.cumulative_link_interference = np.zeros(L)
        self.current_link_queue = np.zeros(self.num_edges)
        self.current_link_capacity = self.bandwidth_edge_list.copy()
        self.num_sps = np.zeros(self.num_edges // 2)
        self.trx_power = self._init_transmission_power()

        # l for link

        if self.received_interference_map is not None:
            for l1 in self.received_interference_map:
                for l2 in self.received_interference_map[l1]:
                    self.interference_map[self.eids[l1], self.eids[l2]] = self.received_interference_map[l1][l2]
                    self.interference_map[self.eids[l2], self.eids[l1]] = self.interference_map[
                        self.eids[l1], self.eids[l2]]

        else:
            for l1 in range(L):
                for l2 in range(l1 + 1, L):
                    r = np.linalg.norm(self.link_pos[l1] - self.link_pos[l2]) * 1e1  # distance [km]
                    if r > sys.float_info.epsilon:
                        self.interference_map[l1, l2] = self.trx_power[l1] / (r ** 2)
                        self.interference_map[l2, l1] = self.trx_power[l2] / (r ** 2)

    def __create_graph(self):
        """
        Create communication graph
        Edges and nodes contains metadata of the network's state
        """
        # create graph
        G = nx.from_numpy_array(self.adjacency_matrix, create_using=nx.DiGraph)
        # assign attributes
        self.graph = G
        self.nodes = list(G.nodes)
        self.num_nodes = G.number_of_nodes()
        self.num_edges = G.number_of_edges()

        # calc interference_map
        self.gen_edge_data()
        self.init_edge_data()

    def show_graph(self, show_fig=True):
        """ draw global graph"""
        plt.figure()
        nx.draw_networkx(self.graph, self.graph_pos, with_labels=False, node_color="tab:blue", node_size=50)
        plt.axis('off')  # suits networkx 2.7.1
        if show_fig:
            plt.show()

    def __init_links(self):
        """init links at env.reset()"""
        self.current_link_interference = np.zeros_like(self.current_link_interference)
        self.cumulative_link_interference = np.zeros_like(self.cumulative_link_interference)
        self.current_link_capacity = self.bandwidth_edge_list.copy()
        self.current_link_queue = np.zeros_like(self.current_link_queue)
        self.last_action = np.zeros_like(self.last_action)
        self.flows_delay = np.zeros_like(self.flows_delay)
        self.flows_rate = np.array(
            [np.min(f + [self.flows[i]["packets"]]) for i, f in enumerate(self.flows_bottleneck)])  # bottleneck is a list of min capacity in each path for each flow
        self.num_sps = np.zeros_like(self.num_sps)
        self.trx_power = self._init_transmission_power()

    def _init_transmission_power(self):
        L = self.num_edges // 2
        power_mode = self.kwargs.get('trx_power_mode', 'equal')
        assert power_mode in ('equal', 'rayleigh', 'rayleigh_gain', 'steps'), f'Invalid power mode. got {power_mode}'
        channel_coeff = np.ones(L)
        channel_gain = np.ones(L)
        if 'rayleigh' in power_mode:
            channel_coeff = np.random.rayleigh(scale=self.kwargs.get('rayleigh_scale', 1), size=L)
        if 'gain' in power_mode:
            channel_gain = self.kwargs.get('channel_gain', np.random.uniform(low=0.5, high=1, size=L)) * np.ones(L)
        p_max = self.kwargs.get('max_trx_power', 1) * np.ones(L)
        trx_power = channel_gain * np.minimum(p_max, 1 / channel_coeff)  # P_l
        if power_mode == 'steps':
            rng = np.max(self.links_length) - np.min(self.links_length)

            trx_power = np.ones(L)
            trx_power[np.where(self.links_length < rng * 1 / 3)] = 1/3
            trx_power[np.where((self.links_length >= rng * 1 / 3) & (self.links_length < rng * 2 / 3))] = 2/3
        return trx_power

    def __calc_possible_actions(self) -> object:
        """ store possible routs into self.possible_actions for each flow"""
        for i, flow in enumerate(self.flows):
            # check if not already calculated:
            if (flow["source"], flow["destination"]) in self.path_bank:
                self.possible_actions[i] = self.path_bank[(flow["source"], flow["destination"])]
            elif (flow["destination"], flow["source"]) in self.path_bank:
                paths = self.path_bank[(flow["destination"], flow["source"])]
                reversed_paths = [list(reversed(p)) for p in paths]
                self.possible_actions[i] = reversed_paths
            else:
                # 1. create new undirected graph with unit weights
                new_graph = self.graph.to_undirected()
                for u, v, attr in new_graph.edges(data=True):
                    attr['weight'] = 1

                # 2. calc k possible routs
                paths = get_k_paths(G=new_graph,
                                    s=flow["source"],
                                    d=flow["destination"],
                                    k=self.k)

                # 3.
                while len(paths) < self.k:
                    paths.append(paths[-1])

                # 4. store valid paths to
                self.possible_actions[i] = paths

                # 5. add to path bank
                self.path_bank[(flow["source"], flow["destination"])] = paths

    def calc_bottleneck(self, path, return_idx=False):
        """
        returns the min capacity(bandwidth) of a link in a path
        """
        path_capacities = [self.current_link_capacity[self.eids[path[j], path[j + 1]]] for j in range(len(path) - 1)]
        idx = np.argmin(path_capacities)
        min_val = path_capacities[idx]
        if return_idx:
            return min_val, idx
        return min_val
    def __calc_flows_bottleneck(self) -> object:
        bank = [[self.calc_bottleneck(path) for path in r] for r in self.possible_actions]
        return bank

    def get_delay_data(self, action_idx=True):
        if action_idx:
            routs = self.get_routs(sorted(self.allocated, key=lambda x: x[0]))
        else:
            routs = [a[1] for a in self.allocated]
        per_flow_delay_above_optimal = [d - (len(r) - 1) for d, r in zip(self.flows_delay, routs)]
        delay_data = {
            "mean_delay": np.mean(self.flows_delay),
            "delay_per_flow": self.flows_delay,
            "above_optimal_delay_per_flow": per_flow_delay_above_optimal,
            "mean_delay_above_optimal": np.mean(per_flow_delay_above_optimal),
            "total_excess_delay": sum(per_flow_delay_above_optimal)
        }
        return delay_data

    def get_rates_data(self) -> object:
        rates_data = {
            "rate_per_flow": self.flows_rate,
            "sum_flow_rates": np.sum(self.flows_rate),
            "sum_log_flow_rates": np.sum(np.log(np.maximum(1, self.flows_rate))),
        }
        return rates_data

    def get_state_space(self):
        """ return possible actions indices"""
        return [[i for i, _ in enumerate(a)] for a in self.possible_actions]

    def set_direction(self, direction):
        self.direction = direction

    def set_tf_env(self, state=False):
        self.tf_env = state

    def edge_list_to_adj_mat(self, lst):
        mat = np.zeros((self.num_nodes, self.num_nodes))
        for eid, l in enumerate(lst):
            u, v = self.id_to_edge[eid]
            mat[u, v] = l
            mat[v, u] = l
        return mat

    def __get_observation(self) -> object:
        """ returns |V|x|V|xd matrix representing the graph """

        # interference
        interference = self.edge_list_to_adj_mat(self.cumulative_link_interference)
        # capacity
        normalized_capacity = self.edge_list_to_adj_mat(self.current_link_capacity)
        if self.normalize_capacity:
            normalized_capacity = np.divide(normalized_capacity, self.bandwidth_matrix,
                                            out=np.zeros_like(normalized_capacity), where=self.bandwidth_matrix != 0)
        normalized_capacity *= self.adjacency_matrix
        adj_matrix = np.stack([interference,
                               normalized_capacity,
                               self.last_action], axis=-1)

        edges = np.array(self.graph.edges)

        allocated = [False] * len(self.flows)
        for i, a in enumerate(self.allocated):
            allocated[a[0]] = True

        allocated = [a[0] for a in self.allocated]
        free_actions = list(set(range(len(self.flows))) - set(allocated))
        free_paths = []
        free_paths_idx = []
        demand = []
        for a in free_actions:
            p = self.possible_actions[a]  # possible paths for flow ('a' is index of a flow)
            free_paths_idx += [[a, k] for k in range(len(p))]
            free_paths += p
            demand += [self.flows[a]["packets"] for k in p]

        # demand
        normalized_demand = np.array(demand).astype(np.float32) / self.max_demand

        if self.tf_env:
            return self.__get_tf_state()

        return adj_matrix, edges, free_paths, free_paths_idx, normalized_demand

    def __get_tf_state(self):
        """
        :return: state for DQN+GNN agent (tf)
        """
        norm_capacity = (np.array(self.current_link_capacity) - self.max_capacity // 2) / self.max_capacity
        betweenes = np.array(self.num_sps) / (((2.0 * self.num_nodes * (self.num_nodes - 1) * self.k) + 0.00000001))

        last_action = np.zeros((self.num_edges // 2, 3))
        for i, u in enumerate(self.last_action):
            for j, v in enumerate(u):
                if v:
                    last_action[self.eids[(i, j)]][0] = 1

        obs = np.concatenate([
            norm_capacity.reshape(-1, 1),
            betweenes.reshape(-1, 1),
            last_action,
            np.zeros((self.num_edges // 2, 15))
        ], axis=1)

        gids = np.array([0] * (self.num_edges // 2))

        return obs, gids, np.array(self.firsts), np.array(self.seconds), self.num_edges // 2

    def reset(self):
        """ reset environment """
        self.history = []
        self.__init_links()
        self.prev_reward = None
        self.allocated = []
        observation = self.__get_observation()
        return observation

    def __update_interference(self, s, d):
        """ update interference due to transmission s->d, effects all edges except (s->*) and (d->s)
            Interference is calculated by 1/(r**2), where r is the distance between two *links*
            Capacity of each link is effected:  capacity = bandwidth*log2(1+SINR) assuming unit transmission power
        """

        trx_power = self.trx_power[self.eids[s, d]]  # P_l
        self.current_link_interference += self.interference_map[self.eids[s, d]]  # I_l for very l
        sinr = trx_power / (self.current_link_interference + 1)  # SINR_l
        self.current_link_capacity = np.minimum(self.bandwidth_edge_list,
                                                np.maximum(1, np.floor(self.bandwidth_edge_list * np.log2(1 + sinr))))

    def __single_local_step(self, u, v, packets):
        """send packet (destined to d) from node u to node v

        @param u: node sending the packet
        @param v: node receiving the packet
        @param packets: list of number of packets to be sent on each flow number of flows that want to send through the link (u,v)
        """

        eid = self.eids[(u, v)]
        link_attr = {"capacity": self.current_link_capacity[eid],
                     "interference": self.current_link_interference[eid]}

        # 1. calc how many (local) time-steps needed to transmit the necessary packets
        link_mac = link_queue_history_using_mac_protocol(link_attr["capacity"], packets)
        num_transmissions = len(link_mac)
        num_transmissions_per_flow = np.count_nonzero(link_mac, axis=0)
        flow_rates = calc_transmission_rate(link_mac)
        assert len(flow_rates) == len(
            packets), f"rates: {flow_rates}, packets: {packets}, capacity: {link_attr['capacity']}"

        # 2. add packets to link's queue
        self.current_link_queue[(eid * 2 - int(u < v)) % self.num_edges] += sum(packets)

        # 3. reward depends on capacity of the link
        # TODO: update reward mechanism
        capacity_reduction = (self.bandwidth_edge_list[eid] - link_attr['capacity']) / self.bandwidth_edge_list[eid]
        reward = 1 * capacity_reduction + \
                 1 * link_attr['interference']

        if self.direction == "maximize":
            reward *= -1

        return reward, num_transmissions, num_transmissions_per_flow, flow_rates

    def __global_step_helper(self, action, active_flows, update_delay=False):
        """send packet (destined to d) from node u to node v

        @param action: |F|x2 matrix, rows=flows, columns=(u,v)
        """
        reward = 0

        action_dict = {}
        for a in action:
            idxs = [idx for idx, b in enumerate(action) if b == a]  # all occurrences of link (u,v)
            packets = [self.flows[active_flows[i]]["packets"] for i in
                       idxs]  # list of packets want to be sent over link (u,v)
            sorted_packets = [(idxs[j], packets[j]) for j, _ in enumerate(packets)]
            sorted_flows = [active_flows[x[0]] for x in sorted_packets]
            sorted_packets = [x[1] for x in sorted_packets]
            action_dict[str(a)] = {"link": a, "packets": sorted_packets, "flows": sorted_flows}

        # 1. transmit on one link for all flows and update interference on the links
        for a in action_dict.values():
            u = a["link"][0]
            v = a["link"][1]

            # update links interference due to transmission u->v
            self.__update_interference(u, v)

        # 2. calculate rewards (influence on others, want to *minimize*)
        transmission_durations = []
        for a in action_dict.values():
            u, v = a["link"]

            # reward
            r, duration, num_transmissions_per_flow, flow_rates = self.__single_local_step(u, v, a["packets"])
            reward += r
            transmission_durations.append(duration)

            # metrics
            for i, f in enumerate(a['flows']):
                # update flows delay
                if update_delay:
                    self.flows_delay[f] += num_transmissions_per_flow[i]

                # update flow rate
                self.flows_rate[f] = np.min(
                    [self.flows_rate[f], flow_rates[i]])  # if self.flows_rate[f] != 0 else flow_rates[i]

        reward += np.max(transmission_durations) * 0.1

        # 3. reset & save interferences for next global step
        self.cumulative_link_interference += self.current_link_interference
        self.current_link_interference = np.zeros_like(self.current_link_interference)
        self.current_link_capacity = self.bandwidth_edge_list.copy()

        return reward

    def __simulate_global_step(self, action, eval_path=False):
        """ simulate transmission of all flows from src->dst and get reward """
        self.allocated.append(action)

        # all routs that are currently allocated in the network
        if not eval_path:
            paths = [self.possible_actions[a[0]][a[1]] for a in sorted(self.allocated, key=lambda x: x[0])]
        else:
            paths = [a[1] for a in sorted(self.allocated, key=lambda x: x[0])]

        # update betweenes values
        p = paths[-1]
        for i in range(len(p) - 1):
            self.num_sps[self.eids[(p[i], p[i + 1])]] += 1

        max_path_length = max([len(p) for p in paths])

        reward = 0
        influence_reward = 0
        for i in range(max_path_length - 1):
            step_action = []  # |F|x2 vector indicates link to transmit on for each flow
            active_flows = []
            for j, p in enumerate(paths):
                if i < len(p) - 1:
                    step_action.append([p[i], p[i + 1]])
                    active_flows.append(self.allocated[j][0])
            # reward for influence on others, want to *minimize*
            influence_reward += self.__global_step_helper(step_action, active_flows,
                                                          update_delay=len(paths) == len(self.flows))

        # self reward (want to *maximize*)
        current_flow, current_rout = action
        rate_reward = self.flows_rate[current_flow] / self.flows[current_flow]['packets']
        # inverse reward for gradient ascent / decent
        if self.direction == "maximize":
            rate_reward *= -1

        alpha = self.reward_balance
        reward = alpha * rate_reward + (1 - alpha) * influence_reward

        # introduce selected action into the graph
        p = self.possible_actions[action[0]][action[1]] if not eval_path else action[1]
        for i in range(len(p) - 1):
            self.last_action[p[i], p[i + 1]] = 1

        if self.prev_reward is not None:
            tmp = reward
            reward = reward - self.prev_reward
            self.prev_reward = tmp
        else:
            self.prev_reward = reward

        return -reward

    def step(self, action, eval_path=False):
        """
        action = (flow_idx, path_idx) if not eval_path (default) else (flow_idx, path)
        :return: reward, routs
        """

        # simulate transmission of all flows from src->dst and get reward
        reward = self.__simulate_global_step(action, eval_path=eval_path)

        # next state
        next_state = self.__get_observation()

        return next_state, reward

    def eval(self, action):
        """
        simulate transmission of all flows from src->dst and get reward

        :param action: list of indices of the paths to allocate, one for each flow
        :return: reward for action
        """

        # simulate transmission of all flows from src->dst and get reward
        reward = self.__simulate_global_step(action)

        return reward

    def eval_all(self, actions):
        """

        :param actions: list of tuples (flow, path)
        :return: accumulated reward
        """
        self.reset()
        rewards = [self.eval((i, a)) for i, a in enumerate(actions)]
        return sum(rewards)

    def get_routs(self, actions):
        if isinstance(actions[0], (list, tuple)):
            return [self.possible_actions[a[0]][a[1]] for a in actions]
        else:
            return [self.possible_actions[i][a] for i, a in enumerate(actions)]

    def rates_objective(self, a):
        self.reset()
        self.eval_all(a)
        return np.sum(self.get_rates_data()['sum_flow_rates'])


if __name__ == "__main__":
    """sanity check for env and reward"""

    # number of nodes
    N = 9

    # Adjacency matrix
    # create 3x3 mesh graph
    A = np.array([[0, 1, 0, 1, 0, 0, 0, 0, 0],
                  [1, 0, 1, 0, 1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 1, 0, 0, 0],
                  [1, 0, 0, 0, 1, 0, 1, 0, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1, 0],
                  [0, 0, 1, 0, 1, 0, 0, 0, 1],
                  [0, 0, 0, 1, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1, 0, 1, 0, 1],
                  [0, 0, 0, 0, 0, 1, 0, 1, 0]])
    P = [(0, 0), (0, 1), (0, 2),
         (1, 0), (1, 1), (1, 2),
         (2, 0), (2, 1), (2, 2)]

    # capacity matrix
    C = 100 * np.ones((N, N))

    # interference matrix
    I = np.ones((N, N)) - np.eye(N)

    # number of paths to choose from
    action_size = 4

    # flow demands
    F = [
        {"source": 0, "destination": 8, "packets": 90, "time_constrain": 10},
        {"source": 1, "destination": 6, "packets": 90, "time_constrain": 10}
    ]

    env = GraphEnvPower(adjacency_matrix=A,
                        bandwidth_matrix=C,
                        interference_matrix=I,
                        flows=F,
                        node_positions=P,
                        k=action_size,
                        reward_balance=0.2)

    adj_matrix, edges, free_paths, free_paths_idx, _ = env.reset()
    possible_actions = free_paths.copy()
    pprint(free_paths)
    best_score = -sys.maxsize
    best_routs = []
    best_action = -1
    for i in range(action_size):
        for j in range(action_size):
            env.reset()
            reward = 0
            a0 = [0, i]
            next_state, r = env.step(action=a0)
            adj_matrix, edges, free_paths, free_paths_idx, norm_demand = next_state
            reward += r
            a1 = [1, j]
            next_state, r = env.step(action=a1)
            adj_matrix, edges, free_paths, free_paths_idx, norm_demand = next_state

            reward += r
            routs = [possible_actions[i], possible_actions[j + action_size]]
            print(
                f"action {[i, j]} with routs {routs} -> reward: {reward} (rates: {env.get_rates_data()['rate_per_flow']}, delay: {env.get_delay_data()['delay_per_flow']})")

            if reward > best_score:
                best_score = reward
                best_routs = routs.copy()
                best_action = [i, j]

    print("*****************************")
    print("Optimal Solution:")
    print(f"action {best_action} with routs {best_routs} -> reward: {best_score}")
