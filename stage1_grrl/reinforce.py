import numpy as np
from torch.distributions import Categorical
import torch
import random
import math


class Reinforce:
    def __init__(self, model, config, optimizer, tb_logger, with_baseline=True):
        super().__init__()
        self.device = torch.device("cpu")  # torch.device("cuda" if torch.cuda.is_available() else 'cpu')

        self.policy = model.to(self.device)
        self.config = config
        self.eps = np.finfo(np.float32).eps.item()
        self.with_baseline = with_baseline
        self.optimizer = optimizer
        self.tb_logger = tb_logger

    def _run_episode(self, env, mode="sample", calc_baseline=True):
        rewards, log_probs = [], []
        state, ep_reward = env.reset(), 0
        for step in range(self.config.num_flows):
            action, log_p = self._select_action(state, method=mode)
            state, reward = env.step(action)
            rewards.append(reward)
            log_probs.append(log_p)
            ep_reward += reward

        # eval baseline
        if calc_baseline:
            baselines = self.best_random_search(env,
                                                num_trials=self.config.num_baseline_trials,
                                                num_steps=self.config.num_flows)

        else:
            baselines = None

        return rewards, log_probs, baselines

    def _select_action(self, state, method):
        adj_matrix, edges, free_paths, free_paths_idx, demand = state
        adj_matrix = torch.from_numpy(adj_matrix).to(self.device).float()
        edges = torch.from_numpy(edges).to(self.device)
        demand = torch.from_numpy(demand).to(self.device)

        probs = self.policy(adj_matrix, edges, free_paths, demand)
        try:
            m = Categorical(probs.view((1, -1)))
            if method == 'sample':
                action = m.sample()
            else:
                action = m.probs.argmax()
        except ValueError as e:
            print(e)
            return free_paths_idx[np.random.uniform(0, len(probs))], 0
        return free_paths_idx[action], m.log_prob(action)

    def _calc_return(self, rewards, normalize=True):
        R = 0
        returns = []
        # discounted rewards
        for r in rewards[::-1]:
            R = r + self.config.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).float()
        if normalize:
            returns = (returns - returns.mean()) / (returns.std() + self.eps)
        return returns

    def _calc_loss(self, rewards, log_probs, baseline=None):
        policy_loss = []

        # discounted rewards
        returns = self._calc_return(rewards, normalize=self.config.norm_rewards)

        # discounted baseline
        baseline_returns = self._calc_return(baseline, normalize=self.config.norm_rewards)

        # loss
        for log_prob, R, B in zip(log_probs, returns, baseline_returns):
            policy_loss.append(-log_prob * (R - B))
        policy_loss = torch.cat(policy_loss).sum()
        return policy_loss

    def train(self, env, episode):
        self.policy.train()

        # Run episode
        rewards, log_probs, baselines = self._run_episode(env)

        # Calculate loss
        loss = self._calc_loss(rewards, log_probs, baselines)

        self.optimizer.zero_grad()
        # Perform backward pass
        loss.backward()

        # Clip gradient norms
        self.clip_grad_norms(self.optimizer.param_groups, self.config.max_grad_norm)

        self.optimizer.step()

        # Logging
        self._log_values(episode, log_probs, rewards, baselines)

    def eval(self, env, mode="sample"):
        self.policy.eval()
        # Run episode
        rewards, _, _ = self._run_episode(env, mode=mode, calc_baseline=False)
        return np.sum(rewards)

    def get_model(self):
        return self.policy

    def _log_values(self, episode, log_likelihood, rewards, baselines):
        cost = np.sum(rewards)
        baseline_cost = np.sum(baselines)

        # Log values to screen
        print(f'\nEpisode: {episode}, cost: {cost}, baseline: {baseline_cost}')

        # Log values to tensorboard
        self.tb_logger.log_value('cost', cost, episode)
        self.tb_logger.log_value('rl_better?/train', int(cost > baseline_cost), episode)

        self.tb_logger.log_value('loss/negative log-likelihood',
                                 -np.mean([ll.cpu().detach().numpy()[0] for ll in log_likelihood]), episode)

    @staticmethod
    def random_episode(env, num_steps):
        state = env.reset()
        rewards = []
        for step in range(num_steps):
            adj_matrix, edges, free_paths, free_paths_idx, demand = state
            action = random.sample(free_paths_idx, 1)[0]
            state, r = env.step(action)
            rewards.append(r)
        return rewards, np.sum(rewards)

    def best_random_search(self, env, num_trials, num_steps):
        best_score = -np.inf
        best_rewards = []
        for i in range(num_trials):
            rewards, score = self.random_episode(env, num_steps)
            if score > best_score:
                best_score = score
                best_rewards = rewards
        return best_rewards

    @staticmethod
    def clip_grad_norms(param_groups, max_norm=math.inf):
        """Clips the norms for all param groups to max_norm and returns gradient norms before clipping
        """
        grad_norms = [
            torch.nn.utils.clip_grad_norm_(
                group['params'],
                max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
                norm_type=2
            )
            for group in param_groups
        ]
        grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
        return grad_norms, grad_norms_clipped
