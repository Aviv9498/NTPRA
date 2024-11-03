"""
implementation of genetic algorithm
inspired by from https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/
"""
import numpy as np
import random

from numpy.random import randint
from numpy.random import rand


class GeneticAlgoritm:
    def __init__(self,
                 n_actions=4,
                 num_selections=100,
                 n_flows=5,
                 n_iter=200,
                 n_pop=1000,
                 r_cross=0.9,
                 r_mut=None):
        self.n_actions = n_actions
        self.num_selections = num_selections
        self.n_flows = n_flows
        self.n_iter = n_iter
        self.n_pop = n_pop
        self.r_cross = r_cross
        self.r_mut = r_mut if r_mut else 1 / self.n_flows

    # tournament selection
    def selection(self, pop, scores):
        # first random selection
        selection_ix = randint(len(pop))
        for ix in randint(0, len(pop), self.num_selections - 1):
            # check if better (e.g. perform a tournament)
            if scores[ix] < scores[selection_ix]:
                selection_ix = ix
        return pop[selection_ix]

    # crossover two parents to create two children
    def crossover(self, p1, p2, r_cross):
        # children are copies of parents by default
        c1, c2 = p1.copy(), p2.copy()
        # check for recombination
        if rand() < r_cross:
            # select crossover point that is not on the end of the string
            pt = randint(1, len(p1) - 2)
            # perform crossover
            c1 = p1[:pt] + p2[pt:]
            c2 = p2[:pt] + p1[pt:]
        return [c1, c2]

    # mutation operator
    def mutation(self, bitstring, r_mut):
        for i in range(len(bitstring)):
            # check for a mutation
            if rand() < r_mut:
                # change rout
                bitstring[i] = randint(low=0, high=self.n_actions)

    # genetic algorithm
    def run(self, objective, n_flows=None, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if n_flows is not None:
            self.n_flows = n_flows
            self.r_mut = 1 / self.n_flows

        # initial population of random bitstring
        pop = [randint(low=0, high=self.n_actions, size=self.n_flows).tolist() for _ in range(self.n_pop)]
        # keep track of best solution
        try:
            best, best_eval = pop[0], objective(pop[0])
        except:
            print('bp')
        # enumerate generations
        for gen in range(self.n_iter):
            # evaluate all candidates in the population
            scores = [objective(c) for c in pop]
            # check for new best solution
            for i in range(self.n_pop):
                if scores[i] < best_eval:
                    best, best_eval = pop[i], scores[i]
            # select parents
            selected = [self.selection(pop, scores) for _ in range(self.n_pop)]
            # create the next generation
            children = list()
            for i in range(0, self.n_pop, 2):
                # get selected parents in pairs
                p1, p2 = selected[i], selected[i + 1]
                # crossover and mutation
                for c in self.crossover(p1, p2, self.r_cross):
                    # mutation
                    self.mutation(c, self.r_mut)
                    # store for next generation
                    children.append(c)
            # replace population
            pop = children
        return best


if __name__ == "__main__":
    """ example from the internet """

    # objective function
    def onemax(x):
        return -sum(x)


    # define the total iterations
    n_iter = 100
    # bits
    n_bits = 20
    # define the population size
    n_pop = 100
    # crossover rate
    r_cross = 0.9
    # mutation rate
    r_mut = 1.0 / float(n_bits)
    # perform the genetic algorithm search
    ga = GeneticAlgoritm(n_actions=2, num_selections=3, n_flows=n_bits, n_iter=n_iter, n_pop=n_pop, r_cross=r_cross,
                         r_mut=r_mut)
    best, score = ga.run(objective=onemax)
    print('Done!')
    print('f(%s) = %f' % (best, score))
