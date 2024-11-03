from datetime import datetime
import os
import shutil
import numpy as np


def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)


def create_trial_folder(src_folder):
    base_path = os.path.abspath(src_folder)

    # current time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # new trial folder
    trial_path = os.path.join(base_path, timestamp)
    if not os.path.exists(trial_path):
        os.makedirs(trial_path)

    # models sub-folder
    checkpoint_path = os.path.join(trial_path, "models")
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    # TensorBoard sub-folder
    tensorboard_path = os.path.join(trial_path, "tensorboard")
    if not os.path.exists(tensorboard_path):
        os.mkdir(tensorboard_path)

    # used-files sub-folder
    usedfiles_path = os.path.join(trial_path, "used_files")
    diamond_path = os.path.abspath("..")
    print("diamond_path:", diamond_path)  # Add this line for debugging

    # Might be unnecessary to copy full folders

    # shutil.copytree(src=os.path.join(diamond_path, "environment"),
    #                 dst=os.path.join(usedfiles_path, "environment"))
    # shutil.copytree(src=os.path.abspath(os.path.join(diamond_path, "stage1_grrl")),
    #                 dst=os.path.join(usedfiles_path, "stage1_grrl"))
    # shutil.copytree(src=os.path.abspath(os.path.join(diamond_path, "stage2_nb3r")),
    #                 dst=os.path.join(usedfiles_path, "stage2_nb3r"))
    return trial_path, tensorboard_path


def calc_num_params(model):
    nb_param = 0
    for param in model.parameters():
        nb_param += np.prod(list(param.data.size()))
    return nb_param


class Stats:
    def __init__(self, max_len):
        self.avg_queue = []
        self.max_len = max_len

    def log(self, value):
        self.avg_queue.append(value)
        while len(self.avg_queue) > self.max_len:
            self.avg_queue.pop(0)

    def get(self):
        return np.sum(self.avg_queue) / self.max_len