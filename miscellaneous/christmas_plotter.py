import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import pickle
from numba import njit
from formatting import get_project_folder
pathname = get_project_folder()

def plot_fractions():
    # Create figure
    fig, ax = plt.subplots(2, 3, figsize=(20, 8))

    cenH_lengths = [0, 3]
    initial_states = ['active', 'active_unmodified', 'unmodified', 'unmodified_silent', 'silent']
    state_colors = ['b', 'g', 'y', 'orange', 'r']
    markers = ['.', 'o', 's', '*', '+']

    DT = 0.02
    T_TOTAL = 2000000

    # Different cenH lengths
    for i in range(len(cenH_lengths)):
        alpha_1_constants = [1, 5, 1]
        alpha_2_constants = [1, 5, 1]
        alpha_1_list = np.linspace(25, 50, 101) * DT * 0.1
        beta_constants = [1, 1, 5]
        alpha_2 = 0.1 * DT * 50
        beta = 0.004

        # Different parameter scales
        for j in range(len(alpha_1_constants)):
            # Print status
            print(i, j)

            # Set parameters
            alpha_1_list *= alpha_1_constants[j]
            alpha_2 *= alpha_2_constants[j]
            beta *= beta_constants[j]

            for k in range(len(initial_states)):
                fractions = np.empty_like(alpha_1_list)

                for l in range(len(alpha_1_list)):
                    alpha_1 = alpha_1_list[l]
                    # Filename to open
                    open_filename = pathname + f'statistics/julesimulationer/states_time_space_init_state={initial_states[k]}_' \
                                    + f'cenH={cenH_lengths[i]}_N=40_t_total={T_TOTAL}_noise=0.5000_alpha_1={alpha_1:.5f}_' \
                                    + f'alpha_2={alpha_2:.5f}_beta={beta:.5f}_seed=0.pkl'

                    # File object
                    file = glob(open_filename)[0]

                    # Open the file
                    with open(file, 'rb') as f:
                        states_time_space = pickle.load(f)[0]

                    # Compute fraction of silent states in last 10th of simulation
                    array_last_tenth = states_time_space[-int(states_time_space.shape[0] / 10):]

                    # Number of total states in array fraction
                    n_elements = array_last_tenth.shape[0] * array_last_tenth.shape[1]

                    silent_fraction = np.count_nonzero(array_last_tenth == 0) / n_elements
                    fractions[l] = silent_fraction

                # Plot fractions
                ax[i, j].scatter(alpha_1_list, fractions, s=3, alpha=1, c=state_colors[k], marker=markers[k],
                                 label=f'{initial_states[k]}', zorder=k)
                ax[i, j].legend(loc='best')
                ax[i, j].set(ylim=(-0.05, 1.05))

            # Set titles
            if i == 0:
                ax[i, j].set_title(r'$\alpha_2$' + f' = {alpha_2:.3f}, ' + r'$\beta$' + f' = {beta:.3f}', size=14)

            # Set x labels
            if i == 1:
                ax[i, j].set_xlabel(r'$\alpha_1$', size=14)

        # Set y labels
        ax[i, 0].set_ylabel(f'Silent fraction, cenH = {cenH_lengths[i]}', size=14)

    fig.tight_layout()
    plt.show()
    return None

def plot_fractions_std():
    # Create figure
    fig, ax = plt.subplots(2, 3, figsize=(20, 8))

    cenH_lengths = [0, 3]
    initial_states = ['active', 'active_unmodified', 'unmodified', 'unmodified_silent', 'silent']
    state_colors = ['b', 'g', 'y', 'orange', 'r']

    DT = 0.02
    T_TOTAL = 2000000

    # Different cenH lengths
    for i in range(len(cenH_lengths)):
        alpha_1_constants = [1, 5, 1]
        alpha_2_constants = [1, 5, 1]
        alpha_1_list = np.linspace(25, 50, 101) * DT * 0.1
        beta_constants = [1, 1, 5]
        alpha_2 = 0.1 * DT * 50
        beta = 0.004

        # Different parameter scales
        for j in range(len(alpha_1_constants)):
            # Set parameters
            alpha_1_list *= alpha_1_constants[j]
            alpha_2 *= alpha_2_constants[j]
            beta *= beta_constants[j]

            for k in range(len(initial_states)):
                stds = np.empty_like(alpha_1_list)
                for l in range(len(alpha_1_list)):
                    alpha_1 = alpha_1_list[l]
                    # Filename to open
                    open_filename = pathname + f'statistics/julesimulationer/states_time_space_init_state={initial_states[k]}_' \
                                    + f'cenH={cenH_lengths[i]}_N=40_t_total={T_TOTAL}_noise=0.5000_alpha_1={alpha_1:.5f}_' \
                                    + f'alpha_2={alpha_2:.5f}_beta={beta:.5f}_seed=0.pkl'

                    # File object
                    file = glob(open_filename)[0]

                    # Open the file
                    with open(file, 'rb') as f:
                        states_time_space = pickle.load(f)[0]

                    # Compute fraction of silent states in last 10th of simulation
                    array_last_tenth = states_time_space[-int(states_time_space.shape[0] / 10):]
                    silent_fractions = np.count_nonzero(array_last_tenth == 0, axis=1) / 40
                    stds[l] = silent_fractions.std(ddof=1)

                # Plot fractions
                ax[i, j].scatter(alpha_1_list, stds, s=1, c=state_colors[k], label=f'{initial_states[k]}')
                ax[i, j].legend(loc='best')
                ax[i, j].set(ylim=(-0.05, 0.55))

            # Set titles
            if i == 0:
                ax[i, j].set_title(r'$\alpha_2$' + f' = {alpha_2:.3f}, ' + r'$\beta$' + f' = {beta:.3f}', size=14)

            # Set x labels
            if i == 1:
                ax[i, j].set_xlabel(r'$\alpha_1$', size=14)

        # Set y labels
        ax[i, 0].set_ylabel(f'Silent fraction stds, cenH = {cenH_lengths[i]}', size=14)

    fig.tight_layout()
    plt.show()

    return None


@njit
def measure_first_decay_time(fractions, l):
    for i in range(len(fractions)):
        # Checks when the other state goes above 50%
        if l == 0:
            if fractions[i] >= 0.5:
                break
        # Checks when the state goes below 50%
        else:
            if fractions[i] < 0.5:
                break
    return i

def plot_decay_times():
    # Create figure
    fig, ax = plt.subplots(2, 3, figsize=(20, 8))

    cenH_lengths = [0, 3]
    initial_states = ['silent', 'active']
    state_colors = ['r', 'b']

    linestyles = ['-', '--']

    DT = 0.02
    T_TOTAL = 2000000

    # Different cenH lengths
    for i in range(len(cenH_lengths)):
        alpha_1_constants = [1, 5, 1]
        alpha_2_constants = [1, 5, 1]
        alpha_1_list = np.linspace(25, 50, 101) * DT * 0.1
        beta_constants = [1, 1, 5]
        alpha_2 = 0.1 * DT * 50
        beta = 0.004

        # Different parameter scales
        for j in range(len(alpha_1_constants)):
            # Set parameters
            alpha_1_list *= alpha_1_constants[j]
            alpha_2 *= alpha_2_constants[j]
            beta *= beta_constants[j]

            # Different initial states
            for k in range(len(initial_states)):

                # Different conditions for decay time:
                # l = 0: The other state passes 50%
                # l = 1: The state in question goes below 50%
                for l in range(2):
                    decay_times = np.empty_like(alpha_1_list)

                    # Different values for alpha_1
                    for m in range(len(alpha_1_list)):
                        # Print progress
                        print(i,j,k,l,m)
                        alpha_1 = alpha_1_list[m]
                        # Filename to open
                        open_filename = pathname + f'statistics/julesimulationer/states_time_space_init_state={initial_states[k]}_' \
                                        + f'cenH={cenH_lengths[i]}_N=40_t_total={T_TOTAL}_noise=0.5000_alpha_1={alpha_1:.5f}_' \
                                        + f'alpha_2={alpha_2:.5f}_beta={beta:.5f}_seed=0.pkl'

                        # File object
                        file = glob(open_filename)[0]

                        # Open the file
                        with open(file, 'rb') as f:
                            states_time_space = pickle.load(f)[0]

                        # Fractions of the initial state in question
                        silent_fractions = np.count_nonzero(states_time_space == 0, axis=1) / 40
                        active_fractions = np.count_nonzero(states_time_space == 2, axis=1) / 40
                        all_fractions = np.block([[silent_fractions], [active_fractions]])

                        # Measure the first decay time
                        # Checks when the other state passes 50%
                        if l == 0:
                            decay_times[m] = measure_first_decay_time(all_fractions[k-1], l)
                        else:
                            decay_times[m] = measure_first_decay_time(all_fractions[k], l)

                    # Plot fractions
                    ax[i, j].plot(alpha_1_list, 100*decay_times, ls=linestyles[l], c=state_colors[k], label=f'{initial_states[k]}')
                    ax[i, j].legend(loc='best')
                   # ax[i, j].set(ylim=(-0.05, 0.55))

            # Set titles
            if i == 0:
                ax[i, j].set_title(r'$\alpha_2$' + f' = {alpha_2:.3f}, ' + r'$\beta$' + f' = {beta:.3f}', size=14)

            # Set x labels
            if i == 1:
                ax[i, j].set_xlabel(r'$\alpha_1$', size=14)

        # Set y labels
        ax[i, 0].set_ylabel(f'Silent fraction stds, cenH = {cenH_lengths[i]}', size=14)

    fig.tight_layout()
    plt.show()

    return None

plot_fractions()

#plot_fractions_std()

#plot_decay_times()
