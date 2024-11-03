import time
import torch
import random
from utils import *
from environment import generate_env
import copy


def train_episode(alg, model, optimizer, lr_scheduler, episode, tb_logger, opts, stats=None):
    print(f"\nEpisode {episode}, lr={optimizer.param_groups[0]['lr']}")

    # log to tensorboard
    tb_logger.log_value('lr', optimizer.param_groups[0]['lr'], episode)
    for name, param in model.named_parameters():
        tb_logger.log_histogram(name, param.data.cpu(), episode)

    # Generate new environment
    env = generate_env(num_nodes=opts.num_nodes,
                       num_edges=opts.num_edges,

                       num_flows=opts.num_flows,
                       min_flow_demand=opts.min_flow_demand,
                       max_flow_demand=opts.max_flow_demand,

                       num_actions=opts.num_actions,

                       min_capacity=opts.min_capacity,
                       max_capacity=opts.max_capacity,

                       direction=opts.direction,
                       reward_balance=opts.reward_balance,
                       seed=opts.seed + episode * 2 + (opts.ep_for_seed + 1) * int(opts.load_weights),
                       graph_mode=opts.graph_mode)

    start_time = time.time()

    alg.train(env=copy.copy(env), episode=episode)

    epoch_duration = time.time() - start_time
    print("Finished episode {}, took {} s".format(episode, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    if optimizer.param_groups[0]['lr'] > 1e-6:
        lr_scheduler.step()

    if opts.checkpoint_epochs != 0 and ((episode > 0 and episode % opts.checkpoint_epochs == 0) or episode == 1):
        print('Saving model and state...')
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'model': model,
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
            },
            os.path.join(opts.save_dir, 'episode-{}.pt'.format(episode))
        )

    # validate
    if episode % opts.validation_every == 0:
        # create validation batch
        env = generate_env(num_nodes=opts.num_nodes,
                           num_edges=opts.num_edges,
                           num_flows=opts.num_flows,
                           min_flow_demand=opts.min_flow_demand,
                           max_flow_demand=opts.max_flow_demand,
                           num_actions=opts.num_actions,
                           min_capacity=opts.min_capacity,
                           max_capacity=opts.max_capacity,
                           direction=opts.direction,
                           reward_balance=opts.reward_balance,
                           seed=opts.seed + episode * 2 + 1 + (opts.ep_for_seed + 1) * int(opts.load_weights))

        valid_cost_sample = alg.eval(env, mode="sample")
        valid_cost_greedy = alg.eval(env, mode="greedy")
        rl_delay_data = env.get_delay_data()
        rl_rates_data = env.get_rates_data()
        valid_cost = np.max([valid_cost_sample, valid_cost_greedy])
        random_score, _, bl_delay_data, bl_rates_data = best_random_search(env=env, num_trials=opts.val_size,
                                                                           num_steps=opts.num_flows)
        tb_logger.log_value('rl_better?/val_sample', int(valid_cost_sample > random_score), episode)
        tb_logger.log_value('rl_better?/val_greedy', int(valid_cost_greedy > random_score), episode)
        tb_logger.log_value('rl_better?/val_delay',
                            int(rl_delay_data['total_excess_delay'] < bl_delay_data['total_excess_delay']), episode)
        tb_logger.log_value('rl_better?/val_rates',
                            int(rl_rates_data['sum_flow_rates'] > bl_rates_data['sum_flow_rates']), episode)

        tb_logger.log_value('val/avg_reward'.format(episode), valid_cost, episode)
        tb_logger.log_value('val/optimality_gap'.format(episode),
                            (random_score - valid_cost) / random_score, episode)
        tb_logger.log_value('val/random_baseline_reward'.format(episode), random_score, episode)
        tb_logger.log_value('val/abs_gap'.format(episode), int(valid_cost > random_score), episode)

        if stats is not None:
            if episode < opts.checkpoint_epochs:
                stats.log(int(valid_cost_greedy > random_score))
            else:
                if episode % opts.checkpoint_epochs == 0:
                    prev = stats.get()
                    stats.log(int(valid_cost_greedy > random_score))
                    curr = stats.get()
                    if curr > prev:
                        print("saving best model")
                        torch.save(
                            {
                                'model_state_dict': model.state_dict(),
                                'model': model,
                                'optimizer': optimizer.state_dict(),
                                'rng_state': torch.get_rng_state(),
                                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                                'episode': episode,
                            },
                            os.path.join(opts.save_dir, f'best_val.pt')
                        )


def random_episode(env, num_steps):
    state = env.reset()
    rewards = []
    for step in range(num_steps):
        adj_matrix, edges, free_paths, free_paths_idx, demand = state
        action = random.sample(free_paths_idx, 1)[0]
        state, r = env.step(action)
        rewards.append(r)
    delay_data = env.get_delay_data()
    rates_data = env.get_rates_data()
    return rewards, np.sum(rewards), delay_data, rates_data


def best_random_search(env, num_trials, num_steps):
    best_score = -np.inf
    best_rewards = []
    save_delay = -np.inf
    save_rates = -np.inf
    for _ in range(num_trials):
        rewards, score, delay_data, rates_data = random_episode(env, num_steps)
        if score > best_score:
            best_score = score
            best_rewards = rewards
            save_delay = delay_data
            save_rates = rates_data
    return best_score, best_rewards, save_delay, save_rates


class Stats:
    def __init__(self, val_size):
        self.avg_queue = []
        self.max_len = val_size

    def log(self, value):
        self.avg_queue.append(value)
        while len(self.avg_queue) > self.max_len:
            self.avg_queue.pop(0)

    def get(self):
        return np.sum(self.avg_queue) / self.max_len
