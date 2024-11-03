import torch
from torch.distributions import Categorical
import sys
import os
sys.path.insert(0, os.path.join("DIAMOND", "stage1_grrl"))


class GRRL:
    """
    Implementation of the GRRL agent (a.k.a. stage_1)
    """
    def __init__(self, path):
        """
        :param path: path to pre-trained model
        """
        # Set the device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # set path
        self.path = path
        # init model
        self.model = self._load_model()

    def _load_model(self):
        """
        load pre-trained model
        :return: loaded model
        """
        try:
            model = torch.load(self.path)['model']
        except FileNotFoundError:
            sys.path.insert(0, os.path.join("..", "stage1_grrl"))
            model = torch.load(self.path.replace('DIAMOND', '..'), weights_only=True)['model']
        model_path = self.path.replace('DIAMOND', '..')
        absolute_path = os.path.abspath(model_path)
        model.eval()
        model.to(self.device)
        return model

    def _select_action(self, state, free_paths):
        """
        select next path to allocate
        :param state: graph state
        :param free_paths: list of available path for allocation
        :return: action as chosen path index
        """
        # prepare state
        adj_matrix, edges, _, free_paths_idx, demand = state
        adj_matrix = torch.from_numpy(adj_matrix).to(self.device).float()
        edges = torch.from_numpy(edges).to(self.device)
        demand = torch.from_numpy(demand).to(self.device)

        # apply model
        probs = self.model(adj_matrix, edges, free_paths, demand)
        # select greedy action
        m = Categorical(probs.view((1, -1)))
        action = m.probs.argmax()

        return int(action.cpu().detach().numpy())

    def run(self, env, given_actions=None, dead_flow_indices=None):
        """
        run GRRL to get flow allocations
        :param env: environment to interact with
        :param given_actions: Sometimes I want to direct the alg to take my paths for rate calculation,
        :param dead_flow_indices: if for some flows want to specify route
        when not specified, the alg calculates the paths and rate accordingly
        :return: action indices, paths and rewards
        """
        state = env.reset()

        if given_actions is not None:
            # my adaptation if we want to specify paths
            actions = given_actions
            paths = []
            reward = 0
            for step in range(env.num_flows):
                action = actions[step]
                paths.append(env.possible_actions[action[0]][action[1]])
                state, r = env.step(action)
                reward += r

        else:
            # Original func
            actions = []
            paths = []
            reward = 0
            for step in range(env.num_flows):
                # ---------------- # My adding if want to specify path for given flow
                if dead_flow_indices is not None and step in dead_flow_indices:
                    a = 3
                else:
                    a = self._select_action(state, env.possible_actions[step])
                # ---------------- #
                action = [step, a]
                actions.append(action)
                paths.append(env.possible_actions[action[0]][action[1]])
                state, r = env.step(action)
                reward += r

        return actions, paths, reward

        # actions is a list of list where every item is a[0] index of flow, a[1] index of path from possible
        # paths
