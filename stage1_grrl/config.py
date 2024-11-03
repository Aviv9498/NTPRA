import argparse
import torch


def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="Arguments and hyperparameters for training")

    # Data
    parser.add_argument('--num_nodes', type=int, default=20,  # 10,
                        help="Number of nodes in the communication graph")
    parser.add_argument('--num_edges', type=int, default=30,  # 20,
                        help="Number of edges in the communication graph")
    parser.add_argument('--num_actions', type=int, default=4,
                        help="k rout choices for each flow")
    parser.add_argument('--num_flows', type=int, default=30,  # 20,
                        help="Number of flow demands")
    parser.add_argument('--min_flow_demand', type=int, default=10,
                        help='Minimum number of packets to transmit')
    parser.add_argument('--max_flow_demand', type=int, default=200,
                        help='Maximum number of packets to transmit')
    parser.add_argument('--min_capacity', type=int, default=500,
                        help='Minimum number of packets to transmit')
    parser.add_argument('--max_capacity', type=int, default=1000,
                        help='Maximum number of packets to transmit')
    parser.add_argument('--graph_mode', type=str, default='snd_geant',
                        help='graph mode for network type in generate env ')

    # GRRL Model
    parser.add_argument('--in_features', type=int, default=3, help="edge input dim")
    parser.add_argument('--hidden_dim', type=int, default=64, help="edge hidden dim")
    parser.add_argument('--demand_hidden_dim', type=int, default=4,
                        help='Dimension of input demand embedding')
    parser.add_argument('--num_iterations', type=int, default=4, help="num message passing iterations")

    # training
    parser.add_argument('--num_episodes', type=int, default=1000, help="num of train episodes")
    parser.add_argument('--gamma', type=float, default=0.001, help="discount reward factor")
    parser.add_argument('--norm_rewards', type=bool, default=True, help="norm reward to zero mean and unit variance")  # True / False
    parser.add_argument('--num_baseline_trials', type=int, default=100, help="num of random trails")
    parser.add_argument('--reward_balance', type=float, default=0.2,
                        help="a\in[0, 1] balance the reward between cooperative and individual goals with a * self_reward + (1-a) * influence_reward")

    # pre-trained
    parser.add_argument('--load_weights', type=bool, default=False, help="load pretrained model")
    parser.add_argument('--saved_weights_path', type=str, help="pretrained model path",
                        default=r"..\pretrained\model_20221113_212726_480.pt")
    parser.add_argument('--ep_for_seed', type=int, default=700, help="num of episodes that the loaded model was trained on")

    # Training
    parser.add_argument('--lr_model', type=float, default=1e-4,
                        help="Set the learning rate for the actor network, i.e. the main model")
    parser.add_argument('--lr_decay', type=float, default=0.5,
                        help='Learning rate decay per decay step')
    parser.add_argument('--lr_decay_step', type=float, default=500,
                        help='Learning rate decay steps (decay every x decay_step epochs)')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum L2 norm for gradient clipping (0 to disable clipping)')
    parser.add_argument('--seed', type=int, default=123, help='Random seed to use')

    # Misc
    parser.add_argument('--log_step', type=int, default=100,
                        help='Log info every log_step steps')
    parser.add_argument('--run_name', default='run', help='Name to identify the run')
    parser.add_argument('--epoch_start', type=int, default=0,
                        help='Start at epoch # (relevant for learning rate decay)')
    parser.add_argument('--checkpoint_epochs', type=int, default=10,
                        help='Save checkpoint every n epochs (default 1), 0 to save no checkpoints')
    parser.add_argument('--validation_every', type=int, default=1,
                        help='eval model every _ batchs')
    parser.add_argument('--val_size', type=int, default=200,
                        help='Number of epochs back for best validation performance')

    parser.add_argument('--result_path', type=str, default='../training_results',
                        help='root folder to backup')
    parser.add_argument('--direction', type=str, default="minimize", help="objective is maximize/minimize reward")

    opts = parser.parse_args(args)

    opts.use_cuda = torch.cuda.is_available()

    return opts
