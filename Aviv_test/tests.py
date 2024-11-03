import numpy as np
import matplotlib.pyplot as plt
from environment import generate_env
from diamond import DIAMOND
import os
import random


MODEL_PATH = os.path.join("DIAMOND", "pretrained", "model_20221113_212726_480.pt")

alg = DIAMOND(grrl_model_path=MODEL_PATH,
              nb3r_tmpr=1.2,
              nb3r_steps=10
              )

all_rates = np.zeros((500, 4))
for i in range(500):
    seed = 10 + (i * 15)

    new_flows = [{'source': 0, 'destination': 2, 'packets': 200},
                 {'source': 5, 'destination': 10, 'packets': 1500},
                 {'source': 10, 'destination': 2, 'packets': 2500},
                 {'source': 8, 'destination': 1, 'packets': 4500},]

    env = generate_env(num_nodes=20,
                       num_edges=30,
                       num_actions=4,
                       num_flows=5,
                       min_flow_demand=1000,
                       max_flow_demand=2000,
                       min_capacity=100,
                       max_capacity=500,
                       seed=seed,
                       graph_mode='abilene',
                       trx_power_mode='equal',
                       rayleigh_scale=1,
                       max_trx_power=10,
                       channel_gain=1,
                       given_flows=new_flows
                       )
    env.show_graph()

    paths = alg(env, use_nb3r=True)

    rates = env.flows_rate
    all_rates[i] = rates

mean_rates = np.mean(all_rates, axis=0)
print('finished')
