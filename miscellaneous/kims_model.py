from timeit import default_timer as DT

import numpy as np
from numba import njit
import matplotlib.pyplot as plt


r = np.random

N_STATES = 3
N_MONOMERS = 60
t_total = 40000
#alpha = 0.58
alpha=0.59

beta = 1 / 3
F = alpha/(1-alpha)
SCATTER_S = 0.1

state_names = {0: 'M', 1: 'U', 2: 'A'}


def f(N_STATES, N_MONOMERS, t_total, alpha, beta):
    t_initial = DT()

    #states = r.randint(N_STATES, size=N_MONOMERS)
    states = np.zeros(N_MONOMERS, dtype=int)
    states[20:40] = 1
    states[40:] = 2
    print(states)
    ts = np.arange(t_total)
    statistics = np.empty((3, t_total))

    statistics = numba_f(states, N_MONOMERS, t_total, alpha, beta, statistics)

    print(f'Time elapsed: {DT() - t_initial:.2f} s')

    return ts, statistics


@njit
def numba_f(states, N_MONOMERS, t_total, alpha, beta, statistics):
    for t in range(t_total):
        if t % (t_total / 10) == 0:
            print(t)
        for i in range(N_MONOMERS):

            # Monomer on which to attempt a change
            n1_index = r.randint(N_MONOMERS)

            # Recruited conversion
            rand_alpha = r.rand()
            if rand_alpha < alpha:
                #print('Recruited')
                #print(rand_alpha)
                # Other monomer
                # Ensure that it is a different monomer
                while True:
                    n2_index = r.randint(N_MONOMERS)
                    if n2_index != n1_index:
                        break

                # If the n2 state is U, do not perform any changes
                if states[n1_index] < states[n2_index] and states[n2_index] != 1:
                    states[n1_index] += 1
                elif states[n1_index] > states[n2_index] and states[n2_index] != 1:
                    states[n1_index] -= 1

            # Noisy conversion
            else:
                rand_beta = r.rand()
                if rand_beta < beta:
                    #print('Noisy')
                    #print(rand_beta)

                    if states[n1_index] == 0:
                        states[n1_index] += 1
                    elif states[n1_index] == 2:
                        states[n1_index] -= 1

                    else:
                        rand = r.rand()
                        if states[n1_index] == 1 and rand < 0.5:
                            states[n1_index] += 1
                        elif states[n1_index] == 1 and rand >= 0.5:
                            states[n1_index] -= 1

        # Statistics
        statistics[0, t] = (states == 0).sum()
        statistics[1, t] = (states == 1).sum()
        statistics[2, t] = (states == 2).sum()

    return statistics


def run(N_STATES, N_MONOMERS, t_total, alpha, beta, SCATTER_S):
        ts, statistics = f(N_STATES, N_MONOMERS, t_total, alpha, beta)

        fig,ax = plt.subplots(figsize=(8,6))

        ax.plot(ts, statistics[0], lw=SCATTER_S, label=state_names[0])
        # ax.plot(ts, statistics[1], lw=SCATTER_S, label=state_names[1])
        # ax.plot(ts, statistics[2], lw=SCATTER_S, label=state_names[2])

        ax.set_title(f'F = {F:.3f}')
        ax.set_xlabel(r'$t$', size=14)
        ax.set_ylabel(r'$N$', size=14)
        ax.legend(loc='best')
        plt.show()


run(N_STATES, N_MONOMERS, t_total, alpha, beta, SCATTER_S)