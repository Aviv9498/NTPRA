from environment import GraphEnvPower as GraphEnv
from environment.utils import *


def _get_random_flows(num_nodes, num_flows, demands=[100], seed=1):
    """
    generates random flows
    :param num_nodes: number of nodes in the communication graph
    :param num_flows: number of flows in the communication graph
    :param demands: list of packets demands for flows to choose from
    :param seed: random seed
    :return: list of flows as (src, dst, pkt)
    """
    random.seed(seed)
    result = []
    for _ in range(num_flows):
        src, dst = random.sample(range(num_nodes), 2)
        f = {"source": src,
             "destination": dst,
             "packets": random.choice(demands)}
        result.append(f)
    return result


def generate_env(num_nodes=10,
                 num_edges=20,

                 num_flows=2,
                 min_flow_demand=100,
                 max_flow_demand=200,

                 num_actions=4,

                 min_capacity=1000,
                 max_capacity=1000,

                 direction="minimize",
                 reward_balance=0.8,
                 seed=37,
                 graph_mode='random',

                 given_flows=None,
                 **kwargs):
    # assert graph_mode.lower() in ['random', 'nsfnet', 'geant']
    assert graph_mode.lower() in ['random', 'nsfnet', 'geant', 'grid', 'irregular_grid_8x8', 'irregular_grid_6x6','abilene', 'snd_geant']

    # 1. create graph
    if graph_mode == 'random':
        adjacency, positions = generate_random_graph(n=num_nodes, e=num_edges, seed=seed)
    elif graph_mode == 'nsfnet':
        adjacency, positions = create_nsfnet_graph()
        num_nodes = 14
    elif graph_mode == 'abilene':
        adjacency, positions = create_abilene_graph()
        num_nodes = 12
    elif graph_mode == 'snd_geant':
        adjacency, positions = create_snd_geant_graph()
        num_nodes = 22
    elif graph_mode == 'geant':
        adjacency, positions = create_geant2_graph()
        num_nodes = 24
    elif graph_mode == 'grid':
        adjacency, positions = gen_grid_graph(n=num_nodes)
        num_nodes = num_nodes ** 2
    elif graph_mode == 'irregular_grid_8x8':
        adjacency, positions = gen_grid_graph(n=8, special_edges_add=[[[1, 0], [2, 2]],
                                                                      [[3, 1], [4, 5]],
                                                                      [[0, 2], [3, 3]],
                                                                      [[4, 1], [6, 2]],
                                                                      [[4, 2], [5, 4]],
                                                                      [[0, 4], [3, 6]],
                                                                      [[4, 3], [7, 4]]
                                                                      ])
        num_nodes = 64
    elif graph_mode == 'irregular_grid_6x6':
        adjacency, positions = gen_grid_graph(n=6, special_edges_remove=[[[1, 0], [1, 1]],
                                                                         [[0, 1], [1, 1]],
                                                                         [[2, 0], [2, 1]],
                                                                         [[2, 1], [3, 1]],
                                                                         [[2, 3], [3, 3]],
                                                                         [[2, 4], [3, 4]],
                                                                         [[2, 5], [3, 5]],
                                                                         [[3, 0], [3, 1]],
                                                                         [[4, 0], [4, 1]],
                                                                         [[4, 1], [5, 1]]
                                                                         ])
        num_nodes = 36

    # 2. create random flows
    delta = 10
    packets = list(range(min_flow_demand, max_flow_demand + delta, delta))

    if given_flows is not None:
        flows = given_flows
    else:
        flows = _get_random_flows(num_nodes=num_nodes, num_flows=num_flows, demands=packets, seed=seed)

    # 3. generate env instance
    # capacity_matrix = np.random.randint(low=min_capacity, high=max_capacity + 1, size=(num_nodes, num_nodes))
    np.random.seed(seed)
    capacity_matrix = np.random.randint(low=min_capacity, high=max_capacity + 1, size=adjacency.shape)

    # interference matrix
    interference_matrix = np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)

    env = GraphEnv(adjacency_matrix=adjacency,
                   bandwidth_matrix=capacity_matrix,
                   interference_matrix=interference_matrix,
                   node_positions=positions,
                   flows=flows,
                   k=num_actions,
                   direction=direction,
                   reward_balance=reward_balance,
                   seed=seed,
                   **kwargs)

    return env
