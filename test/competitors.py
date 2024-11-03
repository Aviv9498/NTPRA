import numpy as np
import random
import networkx as nx
import os
import copy

import tensorflow as tf

from environment.data import generate_env
from test.ga import GeneticAlgoritm

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class RandomBaseline:
    """ Best allocation out of x random trials, Default: x=100. Each trial chooses path for one of k possible paths """

    def __init__(self, num_trials=100):
        self.num_trials = num_trials

    def random_episode(self, env):
        state = env.reset()
        rewards = []
        paths = []
        for step in range(env.num_flows):
            adj_matrix, edges, free_paths, free_paths_idx, demands = state
            action = random.sample(free_paths_idx, 1)[0]
            state, r = env.step(action)
            rewards.append(r)
            paths.append(env.possible_actions[action[0]][action[1]])
        delay_data = env.get_delay_data()
        rates_data = env.get_rates_data()
        return paths, np.sum(rewards), delay_data, rates_data

    def run(self, env, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        best_score = -np.inf
        save_delay = -np.inf
        save_rates = -np.inf
        best_paths = []
        for i in range(self.num_trials):
            paths, score, delay_data, rates_data = self.random_episode(env)
            if score > best_score:
                best_score = score
                best_paths = paths
                save_delay = delay_data
                save_rates = rates_data
        return best_paths, best_score, save_delay, save_rates


class OSPF:
    """ Open Shortest Path First. Allocates best shortest path (not one of the k possible) one flow at a time"""

    @staticmethod
    def _shortest_path(G, s, d):
        path = nx.shortest_path(G, source=s, target=d, weight='weight', method='dijkstra')
        path_weight = sum([G.get_edge_data(path[i], path[i + 1])['weight'] for i in range(len(path) - 1)])
        return path, path_weight

    @staticmethod
    def _process_state(state, env):
        adj_matrix, edges, free_paths, free_paths_idx, demands = state
        normalized_bw = adj_matrix[..., 1]
        free_flows_idx = np.unique(np.array([a[0] for a in free_paths_idx]))
        free_flows = [env.flows[i] for i in free_flows_idx]
        return normalized_bw, free_flows, free_flows_idx

    def _select_action(self, state, env):
        adj_matrix, free_flows, free_flows_idx = self._process_state(state, env)
        # more capacity is better
        adj_matrix[adj_matrix != 0] = 1.01 - adj_matrix[adj_matrix != 0] / env.max_capacity
        G = nx.from_numpy_matrix(adj_matrix)
        best_path = []
        best_path_weight = np.inf
        best_flow_idx = free_flows_idx[0]
        for f, i in zip(free_flows, free_flows_idx):
            path, path_weight = self._shortest_path(G, s=f['source'], d=f['destination'])
            if path_weight < best_path_weight:
                best_path_weight = path_weight
                best_path = path.copy()
                best_flow_idx = i
        return best_flow_idx, best_path   # action is a tuple of -
                                          # (flow idx, list of nodes im the path chosen for the flow)

    def run(self, env, seed=None):
        """

        :param env:
        :param :
        :return:
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        paths = []
        prev_norm = env.normalize_capacity
        env.normalize_capacity = False
        state = env.reset()
        rewards = []
        for step in range(env.num_flows):
            action = self._select_action(state, env)
            paths.append(action[1])
            state, r = env.step(action, eval_path=True)
            rewards.append(r)
        delay_data = env.get_delay_data(action_idx=False)
        rates_data = env.get_rates_data()
        env.normalize_capacity = prev_norm
        return paths, rewards, delay_data, rates_data


class DQN_GNN:
    """
    Adaptation of DQN+GNN by Almasan et.al to our environment
    from source: https://github.com/knowledgedefinednetworking/DRL-GNN
    """

    def __init__(self,
                 path=os.path.join("DIAMOND", "pretrained", "dqn_gnn_weights"),
                 k=4):
        # set path
        self.path = path
        # init model
        try:
            self.model = tf.keras.models.load_model(self.path, compile=False)
        except OSError:
            self.model = tf.keras.models.load_model(self.path.replace('DIAMOND', '..'), compile=False)
        self.model.compile()
        self.k = k

    def _select_action(self, env, step):
        k_envs = [copy.copy(env) for _ in range(len(env.possible_actions[step]))]

        q_vals = []
        for e in k_envs:
            state = e.reset()
            tf_state = [tf.convert_to_tensor(s) for s in state]
            tf_state = [tf.cast(x, tf.int32) if x.dtype == tf.int64 else x for x in tf_state]
            q_val = self.model(*tf_state)
            q_vals.append(q_val[0][0].numpy())

        return np.argmax(q_vals)

    def run(self, env, seed=None):
        """

        :param env:
        :return:
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        actions = []
        paths = []
        env.set_tf_env(state=True)
        env.reset()
        for step in range(env.num_flows):
            action = self._select_action(env, step)
            actions.append(action)
            paths.append(env.possible_actions[step][action].copy())
            env.step([step, action])
        delay_data = env.get_delay_data()
        rates_data = env.get_rates_data()
        env.set_tf_env(state=False)
        return paths, actions, delay_data, rates_data


def eval_current_allocation(env, paths):
    env.reset()
    reward = 0
    for action in enumerate(paths):
        state, r = env.step(action, eval_path=True)
        reward += r
    delay_data = env.get_delay_data(action_idx=False)
    rates_data = env.get_rates_data()
    return rates_data['sum_flow_rates'], np.abs(delay_data['total_excess_delay'])  # np.sum(delay_data['delay_per_flow'])


class IACR:
    """ Interference Aware Cooperative Routing
        https://arxiv.org/pdf/2201.01520.pdf
    """
    def __init__(self, delta=0.5, alpha=0.1):
        self.delta = delta
        self.alpha = alpha  # rates balancing

    @staticmethod
    def _shortest_path(G, s, d):
        path = nx.shortest_path(G, source=s, target=d, weight='weight', method='dijkstra')
        path_weight = sum([G.get_edge_data(path[i], path[i + 1])['weight'] for i in range(len(path) - 1)])
        return path, path_weight

    def _process_state(self, state, env):
        adj_matrix, _, _, free_paths_idx, _ = state
        capacity_matrix = adj_matrix[..., 1]
        capacity_matrix[capacity_matrix != 0] = 1.01 - capacity_matrix[capacity_matrix != 0] / env.max_capacity # more capacity is better
        free_flows_idx = np.unique(np.array([a[0] for a in free_paths_idx]))
        free_flows = [env.flows[i] for i in free_flows_idx]

        received_interference = env.edge_list_to_adj_mat(env.cumulative_link_interference)  # interference on link
        created_interference = env.edge_list_to_adj_mat(np.sum(env.interference_map, axis=0))  # interference created by link
        received_interference /= np.max(received_interference) if np.max(received_interference) > 1e-6 else 1
        created_interference /= np.max(created_interference) if np.max(created_interference) > 1e-6 else 1
        adj_matrix = self.delta * created_interference + (1 - self.delta) * received_interference \
                   + self.alpha * capacity_matrix
        adj_matrix[adj_matrix < 0] = 1e-6

        return adj_matrix, free_flows, free_flows_idx

    def _select_action(self, state, env):
        adj_matrix, free_flows, free_flows_idx = self._process_state(state, env)
        G = nx.from_numpy_matrix(adj_matrix)
        best_path = []
        best_path_weight = np.inf
        best_flow_idx = free_flows_idx[0]
        for f, i in zip(free_flows, free_flows_idx):
            path, path_weight = self._shortest_path(G, s=f['source'], d=f['destination'])
            if path_weight < best_path_weight:
                best_path_weight = path_weight
                best_path = path.copy()
                best_flow_idx = i
        return best_flow_idx, best_path

    def run(self, env, seed=None):
        """

        :param env:
        :param :
        :return:
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        paths = []
        prev_norm = env.normalize_capacity
        env.normalize_capacity = False
        state = env.reset()
        rewards = []
        for step in range(env.num_flows):
            action = self._select_action(state, env)
            paths.append(action[1])
            state, r = env.step(action, eval_path=True)
            rewards.append(r)
        delay_data = env.get_delay_data(action_idx=False)
        rates_data = env.get_rates_data()
        env.normalize_capacity = prev_norm
        return paths, rewards, delay_data, rates_data


class DIAR:
    """ DELAY- AND INTERFERENCE-AWARE ROUTING
        https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8995471
    """
    def __init__(self,
                 n_iter=2,
                 n_pop=100,
                 num_selections=5):
        self.ga = GeneticAlgoritm(n_iter=n_iter, n_pop=n_pop, num_selections=num_selections)

    def _process_state(self, state, env):
        _, _, free_paths, free_paths_idx, _ = state
        free_flows_idx = np.unique(np.array([a[0] for a in free_paths_idx]))
        free_flows = [env.flows[i] for i in free_flows_idx]
        adj_matrix = env.adjacency_matrix

        return adj_matrix, free_flows, free_flows_idx, free_paths, free_paths_idx

    def _select_action(self, state, env):
        adj_matrix, free_flows, free_flows_idx, free_paths, free_paths_idx = self._process_state(state, env)
        test_env = copy.copy(env)
        best_path = []
        best_path_weight = np.inf
        best_flow_idx = free_flows_idx[0]
        for flow_idx in free_flows_idx:
            test_env.reset()
            for path in env.possible_actions[flow_idx]:
                test_env.step((flow_idx, path), eval_path=True)
                delay_data = test_env.get_delay_data(action_idx=False)
                path_weight = max(delay_data['delay_per_flow'][flow_idx], len(path) - 1)
                if path_weight < best_path_weight:
                    best_path_weight = path_weight
                    best_path = path.copy()
                    best_flow_idx = flow_idx
        return best_flow_idx, best_path

    def objective(self, actions):
        self.env.reset()
        self.env.eval_all(actions)
        return np.sum(self.env.get_delay_data()['mean_delay'])

    def run(self, env, seed=None):
        """

        :param env:
        :param :
        :return:
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.env = env

        actions = self.ga.run(objective=self.objective, n_flows=self.env.num_flows, seed=seed)
        try:
            paths = self.env.get_routs(actions)
        except:
            print(seed)
            print('bp')

        rewards = []
        state = env.reset()
        for flow_idx, path in enumerate(paths):
            action = (flow_idx, path)
            state, r = env.step(action, eval_path=True)
            rewards.append(r)
        delay_data = env.get_delay_data(action_idx=False)
        rates_data = env.get_rates_data()
        return paths, rewards, delay_data, rates_data
