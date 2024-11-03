import json
import torch
import torch.optim as optim
from tensorboard_logger import Logger as TbLogger

from config import get_options
from utils import *
from gnn import GNN
from reinforce import Reinforce
from train import train_episode


def run(opts):
    torch.cuda.empty_cache()

    # Set the random seed
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)

    trial_path, tensorboard_path = create_trial_folder(opts.result_path)
    opts.save_dir = os.path.join(trial_path, "models")
    tb_logger = TbLogger(tensorboard_path)
    print(f"saving results to {trial_path}")

    # Save arguments so exact configuration can always be found
    with open(os.path.join(trial_path, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

    # Set the device
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    model = GNN(in_features=opts.in_features,
                hidden_dim=opts.hidden_dim,
                demand_hidden_dim=opts.demand_hidden_dim,
                num_iterations=opts.num_iterations)

    # Compute number of network parameters
    print(model)
    print('Number of parameters: ', calc_num_params(model))

    # Initialize optimizer
    optimizer = optim.Adam([{'params': model.parameters(), 'lr': opts.lr_model}])

    # load pre-trained weights
    if opts.load_weights:
        print("\nloading pre-trained weights")
        checkpoint = torch.load(opts.saved_weights_path)
        try:
            model = torch.load(checkpoint['model'])
        except:
            model.load_state_dict(checkpoint['model_state_dict'])

    # TODO: add new schedulers
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay)

    reinforce = Reinforce(model=model, config=opts, optimizer=optimizer, tb_logger=tb_logger, with_baseline=True)

    stats = Stats(opts.val_size)

    # Start training loop
    for episode in range(opts.num_episodes):
        train_episode(
            alg=reinforce,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            episode=episode,
            tb_logger=tb_logger,
            opts=opts,
            stats=stats)


if __name__ == "__main__":
    run(get_options())
    print('ok')
