import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
import os
import random


def init_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    np.random.RandomState(seed)
    np.random.seed(seed)
    random.seed(seed)


def update_graph_weight_from_path(G: nx.Graph,
                                  P: list,
                                  x=None):
    """
    @param G: graph
    @param P: path in G (list of nodes)
    @param x: factor to be added to graph weights

    @precondition: x must be power of two

    """
    # copy graph to avoid changing the original one
    A = G.copy()

    if x is None:
        x = 2 ** nx.algorithms.distance_measures.diameter(G)

    d = P[-1]  # destination node
    V = set(G.nodes)  # all nodes in G
    U = {d}  # finished nodes
    S = set()  # unfinished nodes

    # add weights to all edges of P and their first neighbors
    for i in range(len(P) - 1):
        u = P[i]
        v = P[i + 1]
        A.get_edge_data(u, v)['weight'] += x
        for j in A.neighbors(u):
            if j != v and j not in P:
                A.get_edge_data(u, j)['weight'] += x / 2
                S.add(j)
        U.add(u)

    # add weights to d's edges
    for j in A.neighbors(d):
        if j not in P:
            A.get_edge_data(d, j)['weight'] += x / 2
            S.add(j)

    x /= 4
    # add weights iteratively to all other edges w.r.t their distance to P
    while U != V and len(S) > 0:
        T = set()
        for u in S:
            for v in A.neighbors(u):
                if v in V - U:
                    A.get_edge_data(u, v)['weight'] += x
                    if v not in S:
                        T.add(v)
            U.add(u)
        x /= 2
        S = T.copy()

    return A


def show_graph(G: nx.Graph):
    """ draw graph"""
    plt.figure()
    # Create positions of all nodes
    pos = nx.spring_layout(G)
    # Draw the graph according to node positions
    nx.draw_networkx(G, pos, with_labels=True, node_color="tab:blue")
    # Create edge labels
    labels = {(u, v): e['weight'] for u, v, e in G.edges(data=True)}
    # Draw edge labels according to node positions
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    plt.axis('off')
    plt.show()


def get_k_paths(G: nx.Graph,
                s: int,
                d: int,
                k: int,
                x=None,
                method='dijkstra'):
    """
    Returns k (shortest) most separated paths in G from s to d
    """
    k_sp = []
    A = G.copy()

    # init weights
    for u, v, attr in A.edges(data=True):
        attr['weight'] = 1

    # all_possible_paths = list(nx.shortest_simple_paths(G, source=s, target=d, weight='weight'))
    gen_path = nx.shortest_simple_paths(G, source=s, target=d, weight='weight')
    # not exactly all possible paths but enough options to choose from, just to speed things
    all_possible_paths = []
    for _ in range(k ** 2):
        try:
            all_possible_paths.append(next(gen_path))
        except StopIteration:
            break

    while len(k_sp) < k and len(k_sp) < len(all_possible_paths):
        try:
            p = nx.shortest_path(A, source=s, target=d, weight='weight', method=method)
        except:
            break
        if p not in k_sp:
            k_sp.append(p)
        else:
            # if the chosen path already exists, try to take one of the other paths (Yen's algorithm)
            sp = nx.shortest_simple_paths(G, source=s, target=d, weight='weight')  # generator, doesn't actually calculates all paths at ones
            next(sp)  # first item the p, so we skip it
            for _ in range(k):
                try:
                    p = next(sp)
                except StopIteration:
                    break
                if p not in k_sp:
                    k_sp.append(p)
                    break
        A = update_graph_weight_from_path(A, p, x)

    return k_sp


def one_link_transmission(c, packets):
    """one transmission on channel with capacity c for flows with packets packets
    @param c: link's capacity
    @param packets: list of flows load (packets)

    returns list of packets remaining to be sent for each flow
    """
    if sum(packets) <= c:
        return [0] * len(packets)

    count = len(packets) - len([p for p in packets if p == 0])
    q = c // count
    r = c % count
    new_packs = []
    for p in packets:
        new_packs.append(p - min(p, q))
        if p > 0:
            r += q - min(p, q)
    i = 0
    while r > 0 and i < len(new_packs):
        if new_packs[i] > 0:
            rem = min(r, new_packs[i])
            new_packs[i] -= rem
            r -= rem
        i += 1
    return new_packs


def link_queue_history_using_mac_protocol(c, packets):
    """
    returns history of transmission for one or more flows on a channel with capacity c
    assuming some MAC protocol exists on the link.
    When multiple agents transmit together on a shared link, they split the capacity until all have been sent

    :param c: link's capacity
    :param packets: list of flows load (packets)
    """
    hist = []
    packs = packets.copy()
    while sum(packs) > 0:  # until all flows have transmitted all their packets
        hist.append(packs)
        packs = one_link_transmission(c, packs)
    return np.array(hist)


def calc_transmission_rate(link_mac):
    """
    calculates the transmission rate (based on bottleneck)
    :param link_mac:
    :return:
    """
    if len(link_mac) == 1:
        return link_mac[0]
    trans = np.stack([link_mac[i] - link_mac[i+1] for i in range(len(link_mac)-1)], axis=0)
    # if len(trans.shape) == 1:
    #     return trans
    return np.min(trans, axis=0)


def generate_random_graph(n, e, seed=None):
    """
    generate random graph in [0,1]^2 contains of n nodes
    :param n: number of nodes in the graph
    :param e: number of edges in the graph
    :return: nx.Graph
    """

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # adjacency matrix
    adjacency = np.zeros((n, n))
    # sample nodes positions in [0,1]^2
    positions = np.random.uniform(low=0, high=1, size=(n, 2))

    # connect nodes via path to make sure the graph is connected
    for i in range(n-1):
        adjacency[i, i+1] = 1
        adjacency[i+1, i] = 1

    # sample random edges
    for _ in range(e-n+1):
        i, j = random.sample(range(n), 2)
        if np.array_equal(adjacency, np.ones((n,n))-np.eye(n)):
            print(f"full graph has {(n ** 2 - n) // 2} ({n ** 2 - n}) edges instead of {e} ({e*2})")
            break
        while adjacency[i, j]:
            i, j = random.sample(range(n), 2)
        adjacency[i, j] = 1
        adjacency[j, i] = 1

    g = nx.from_numpy_matrix(adjacency, create_using=nx.Graph)
    for u, v, attr in g.edges(data=True):
        # set node position
        if g.nodes[u] == {}:
            g.nodes[u]['pos'] = positions[u]
        if g.nodes[v] == {}:
            g.nodes[v]['pos'] = positions[v]

    # return g
    return adjacency, positions


def create_geant2_graph():
    """
    nodes and edges from: https://github.com/knowledgedefinednetworking/DRL-GNN/blob/master/DQN/gym-environments/gym_environments/envs/environment1.py
    """
    Gbase = nx.Graph()
    Gbase.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
    Gbase.add_edges_from(
        [(0, 1), (0, 2), (1, 3), (1, 6), (1, 9), (2, 3), (2, 4), (3, 6), (4, 7), (5, 3),
         (5, 8), (6, 9), (6, 8), (7, 11), (7, 8), (8, 11), (8, 20), (8, 17), (8, 18), (8, 12),
         (5, 8), (6, 9), (6, 8), (7, 11), (7, 8), (8, 11), (8, 20), (8, 17), (8, 18), (8, 12),
         (9, 10), (9, 13), (9, 12), (10, 13), (11, 20), (11, 14), (12, 13), (12, 19), (12, 21),
         (14, 15), (15, 16), (16, 17), (17, 18), (18, 21), (19, 23), (21, 22), (22, 23)])
    A = np.array(nx.to_numpy_array(Gbase))
    A = np.clip(A + A.T, a_min=0, a_max=1)
    pos = nx.spring_layout(Gbase, seed=1234)
    pos = np.stack(list(pos.values()), axis=0)
    return A, pos


def create_nsfnet_graph():
    """
    nodes and edges from: https://github.com/knowledgedefinednetworking/DRL-GNN/blob/master/DQN/gym-environments/gym_environments/envs/environment1.py
    """
    Gbase = nx.Graph()
    Gbase.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    Gbase.add_edges_from(
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 7), (2, 5), (3, 8), (3, 4), (4, 5), (4, 6), (5, 12), (5, 13),
         (6, 7), (7, 10), (8, 9), (8, 11), (9, 10), (9, 12), (10, 11), (10, 13), (11, 12)])
    A = np.array(nx.to_numpy_array(Gbase))
    A = np.clip(A + A.T, a_min=0, a_max=1)
    G = nx.from_numpy_array(A)
    pos = nx.spring_layout(G, seed=124)
    pos = np.stack(list(pos.values()), axis=0)
    return A, pos


def create_abilene_graph():

    """
    node positions from: https://sndlib.put.poznan.pl/home.action?show=/networks.overview.action%3Fframeset
    @return: abilene graph
    """

    Gbase = nx.Graph()
    Gbase.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    Gbase.add_edges_from(
        [(0, 1),
         (1, 4), (1, 5), (1, 11),
         (2, 5), (2, 8),
         (3, 6), (3, 9), (3, 10),
         (4, 6), (4, 7),
         (5, 6),
         (7, 9),
         (8, 11),
         (9, 10)])
    A = np.array(nx.to_numpy_array(Gbase))
    A = np.clip(A + A.T, a_min=0, a_max=1)
    G = nx.from_numpy_array(A)
    # pos = nx.spring_layout(G, seed=124)
    # Manually setting positions based on the provided coordinates
    pos = {
        0: (-84.3833, 33.75),        # ATLAM5
        1: (-85.50, 34.50),          # ATLAng
        2: (-87.6167, 41.8333),      # CHINng
        3: (-105.00, 40.75),         # DNVRng
        4: (-95.517364, 29.770031),  # HSTNng
        5: (-86.159535, 39.780622),  # IPLSng
        6: (-96.596704, 38.961694),  # KSCYng
        7: (-118.25, 34.05),         # LOSAng
        8: (-73.9667, 40.7833),      # NYCMng
        9: (-122.02553, 37.38575),   # SNVAng
        10: (-122.30, 47.60),        # STTLng
        11: (-77.026842, 38.897303)  # WASHng
    }
    pos = np.stack(list(pos.values()), axis=0)
    return A, pos


def create_snd_geant_graph():
    """
    node positions from: https://sndlib.put.poznan.pl/home.action
    @return: abilene graph
    """
    Gbase = nx.Graph()

    # Add nodes to the graph
    Gbase.add_nodes_from(range(22))  # 22 nodes

    # Add edges to the graph (links between nodes)
    Gbase.add_edges_from([
        (0, 2),   # at1.at -> ch1.ch
        (0, 4),   # at1.at -> de1.de
        (0, 9),   # at1.at -> hu1.hu
        (0, 15),  # at1.at -> ny1.ny
        (0, 19),  # at1.at -> si1.si
        (1, 6),   # be1.be -> fr1.fr
        (1, 13),  # be1.be -> lu1.lu
        (1, 14),  # be1.be -> nl1.nl
        (2, 6),   # ch1.ch -> fr1.fr
        (2, 11),  # ch1.ch -> it1.it
        (3, 4),   # cz1.cz -> de1.de
        (3, 16),  # cz1.cz -> pl1.pl
        (3, 20),  # cz1.cz -> sk1.sk
        (4, 6),   # de1.de -> fr1.fr
        (4, 7),   # de1.de -> gr1.gr
        (4, 10),  # de1.de -> ie1.ie
        (4, 11),  # de1.de -> it1.it
        (4, 14),  # de1.de -> nl1.nl
        (4, 18),  # de1.de -> se1.se
        (5, 6),   # es1.es -> fr1.fr
        (5, 11),  # es1.es -> it1.it
        (5, 17),  # es1.es -> pt1.pt
        (6, 13),  # fr1.fr -> lu1.lu
        (6, 21),  # fr1.fr -> uk1.uk
        (7, 11),  # gr1.gr -> it1.it
        (8, 9),   # hr1.hr -> hu1.hu
        (8, 19),  # hr1.hr -> si1.si
        (9, 20),  # hu1.hu -> sk1.sk
        (10, 21), # ie1.ie -> uk1.uk
        (12, 11), # il1.il -> it1.it
        (12, 14), # il1.il -> nl1.nl
        (14, 21), # nl1.nl -> uk1.uk
        (15, 21), # ny1.ny -> uk1.uk
        (16, 18), # pl1.pl -> se1.se
        (17, 21), # pt1.pt -> uk1.uk
        (18, 21)  # se1.se -> uk1.uk
    ])

    # Create adjacency matrix
    A = np.array(nx.to_numpy_array(Gbase))
    A = np.clip(A + A.T, a_min=0, a_max=1)

    # Create the position dictionary for node coordinates
    pos = {
        0: (16.3729, 48.2091),   # at1.at
        1: (4.3518, 50.8469),    # be1.be
        2: (6.1399, 46.2038),    # ch1.ch
        3: (14.4423, 50.0785),   # cz1.cz
        4: (8.6842, 50.1122),    # de1.de
        5: (-3.7033, 40.4167),   # es1.es
        6: (2.351, 48.8566),     # fr1.fr
        7: (23.5808, 37.9778),   # gr1.gr
        8: (15.9644, 45.8071),   # hr1.hr
        9: (19.0936, 47.4976),   # hu1.hu
        10: (-6.2573, 53.3416),  # ie1.ie
        11: (9.19, 45.4642),     # it1.it
        12: (34.8097, 32.0714),  # il1.il
        13: (6.1296, 49.6112),   # lu1.lu
        14: (4.9407, 52.3236),   # nl1.nl
        15: (-73.94384, 40.6698),  # ny1.ny
        16: (16.8874, 52.3963),  # pl1.pl
        17: (-9.1363, 38.7073),  # pt1.pt
        18: (17.8742, 59.3617),  # se1.se
        19: (14.5148, 46.0574),  # si1.si
        20: (17.1297, 48.1531),  # sk1.sk
        21: (-0.1264, 51.5086)   # uk1.uk
    }

    # Stack positions in an array
    pos = np.stack(list(pos.values()), axis=0)

    return A, pos


def gen_grid_graph(n, special_edges_add=None, special_edges_remove=None):
    """
    Generates adjacency matrix for a nxn grid network
    :param n: number of nodes in each edge (total n**2 nodes)
    :param special_edges_add: list(list(tuple)) indicates extra edges to add. example: [[(0,0), (3, 2)]]
    :param special_edges_remove: list(list(tuple)) indicates edges to remove. example: [[(0,0), (0, 1)]]
    :return: adjacency matrix of a mesh grid network
    """
    N = int(n ** 2)
    A = np.zeros((N, N))

    # gen basic grid
    for i in range(n):
        for j in range(n):
            my_id = i * n + j
            if i - 1 >= 0:
                neighbor_id = (i - 1) * n + j
                A[my_id, neighbor_id] = 1
                A[neighbor_id, my_id] = 1
            if i + 1 <= n - 1:
                neighbor_id = (i + 1) * n + j
                A[my_id, neighbor_id] = 1
                A[neighbor_id, my_id] = 1
            if j - 1 >= 0:
                neighbor_id = i * n + (j - 1)
                A[my_id, neighbor_id] = 1
                A[neighbor_id, my_id] = 1
            if j + 1 <= n - 1:
                neighbor_id = i * n + (j + 1)
                A[my_id, neighbor_id] = 1
                A[neighbor_id, my_id] = 1

    # add specified edges
    if special_edges_add is not None:
        for i, j in special_edges_add:
            i_id = i[0] * n + i[1]
            j_id = j[0] * n + j[1]
            A[i_id, j_id] = 1
            A[j_id, i_id] = 1

    # remove specified edges
    if special_edges_remove is not None:
        for i, j in special_edges_remove:
            i_id = i[0] * n + i[1]
            j_id = j[0] * n + j[1]
            A[i_id, j_id] = 0
            A[j_id, i_id] = 0

    # nodes positions in [0,1]^2
    x = np.linspace(0, 1, n)
    positions = np.array([[i, j] for i in x for j in x[::-1]])

    return A, positions


if __name__ == "__main__":
    """ test functions"""

    c = 2
    packets = [5, 3, 2, 0, 4]
    result = one_link_transmission(c, packets)
    print(result)
    history = link_queue_history_using_mac_protocol(c,packets)
    print(history)
    print(calc_transmission_rate(history))