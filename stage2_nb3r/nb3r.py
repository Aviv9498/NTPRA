import random
import numpy as np


def softmax(x, i, b=1):
    return np.exp(x[i] * b) / np.sum(np.exp(x * b))


def pmf(x, b=1):
    return [softmax(x, i, b) for i, _ in enumerate(x)]


def nb3r(objective, state_space, num_iterations, initial_temperature=0.8,
         initial_state=None, verbose=False, seed=None, return_history=False):
    """
    Find a minima for discrete function objective using Distributed Simulated Annealing

    :param return_history:
    :param initial_state:
    :param objective: function to minimize (V(.) in the paper)
    :param state_space: valid values for variables. Nested list: rows-variables(flows), columns- values(paths)
    :param initial_temperature: initial temperature
    :param num_iterations: number of search iteration
    :param verbose: print during training if true
    :param seed: random seed

    :return: list of best params
    """

    # initializations
    np.random.seed(seed=seed)
    random.seed(seed)
    num_flows = len(state_space)

    # generate an initial point
    if initial_state is None:
        state = [np.random.choice(v) for v in state_space]
    else:
        state = initial_state.copy()
    # evaluate the initial point
    state_eval = objective(state)

    best, best_eval = state, state_eval

    if verbose:
        print(f"initial guess: {state}, score: {state_eval}")

    history = [state.copy()]

    # run the algorithm
    for k in range(num_iterations):
        # calk T_k
        temperature = initial_temperature / (k + 1)

        # choose random flow
        flow = np.random.randint(num_flows)

        possible_routs = state_space[flow]
        num_routs = len(possible_routs)

        # build transition probability vector
        probs = np.ones(num_routs) * (1 / num_routs)  # R(x,y) = 1/|N(x)|
        # P_k(x,y) for y in N(x) and y != x
        for i, rout in enumerate(possible_routs):
            if i != state[flow]:
                y = state.copy()
                y[flow] = rout
                probs[i] *= np.exp(-np.maximum(objective(y) - state_eval, 0) / temperature)

        try:
            # make valid probability distribution (softmax)
            probs /= sum(probs)
            # update state
            state[flow] = np.random.choice(possible_routs, p=probs)
        except:
            pass
        # evaluate state
        state_eval = objective(state)

        history.append(state.copy())

        if state_eval < best_eval:
            criterion = 'improved [^]'
            best, best_eval = state, state_eval
        elif state_eval > best_eval:
            criterion = 'random [?]'
        else:
            criterion = 'unchanged [-]'

        if verbose:
            print(f"iteration: {k + 1}. flow: {flow} choice: {state}, score: {state_eval}, criterion: {criterion}")

    if return_history:
        return state, history
    else:
        return state

# state output is a list of N items where every item is the index of path regarding the flow
# [1,3,7,8] first flow took path 1, second flow took path 3 and so on
