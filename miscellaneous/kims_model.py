from timeit import default_timer as DT
import os

import numpy as np
from numba import njit
import matplotlib.pyplot as plt


r = np.random
SEED = 0

N_STATES = 3
N_MONOMERS = 60
t_total = 10000
alpha=0.59

beta = 1 / 3
F = alpha/(1-alpha)
LW = 0.15

state_names = {0: 'M', 1: 'U', 2: 'A'}
    

def f(N_STATES, N_MONOMERS, t_total, alpha, beta):
    t_initial = DT()

    #states = r.randint(N_STATES, size=N_MONOMERS)
    states = np.zeros(N_MONOMERS, dtype=int)
    states[20:40] = 1
    states[40:] = 2

    ts = np.arange(t_total)
    statistics = np.empty((3, t_total))

    statistics = numba_f(states, N_MONOMERS, t_total, alpha, beta, statistics)

    print(f'Time elapsed: {DT() - t_initial:.2f} s')

    return ts, statistics


@njit
def numba_f(states, N_MONOMERS, t_total, alpha, beta, statistics):
    # Set Numba seed
    np.random.seed(SEED)

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
                if (states[n1_index] < states[n2_index]
                        and states[n2_index] != 1):
                    states[n1_index] += 1

                elif (states[n1_index] > states[n2_index]
                      and states[n2_index] != 1):
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


def main(N_STATES, N_MONOMERS, t_total, alpha, beta, LW):
        # Set Numpy seed
        np.random.seed(SEED)

        ts, statistics = f(N_STATES, N_MONOMERS, t_total, alpha, beta)

        fig,ax = plt.subplots(figsize=(4.792, 3.0))

        ax.plot(ts, statistics[0], lw=LW, label=state_names[0], c='r')
        # ax.plot(ts, statistics[1], lw=LW, label=state_names[1])
        ax.plot(ts, statistics[2], lw=LW, label=state_names[2], c='b')

        #ax.set_title(f'F = {F:.3f}')
        ax.set_xlabel(r'$t$')
        ax.set_ylabel('$N$')
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        ax.legend(loc=(0.82,0.26))
        fig.tight_layout()

        # Save figure
        figname = ('../../../Documents/masters_thesis/ThesisPaperFigures/'
                   + 'dodd_results.pdf')

        plt.savefig(figname)

        plt.show()

if __name__ == '__main__':
    main(N_STATES, N_MONOMERS, t_total, alpha, beta, LW)