import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
# from mayavi import mlab
# from mayavi.mlab import *

def plot_final_state(open_filename):
    with open(open_filename, 'rb') as f:
        x_plot, y_plot, z_plot, states, state_colors, state_names, plot_dim = pickle.load(f)

    # Create figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the different states
    for i in range(len(states)):
        ax.scatter(x_plot[states[i]], y_plot[states[i]], z_plot[states[i]], s=5, c=state_colors[i])

    # Plot chain line
    all_condition = torch.ones_like(states[0], dtype=torch.bool)

    ax.plot(x_plot[all_condition], y_plot[all_condition], z_plot[all_condition],
            marker='o', ls='solid', markersize=1, c='k', lw=0.7, label='Final state')

    for i in range(len(state_colors)):
        ax.scatter([],[],c=state_colors[i],label=state_names[i])

    ax.legend(loc='upper left')
    ax.set_title(f'No. of nucleosomes = {len(x_plot)}', size=16)
    ax.set(xlim=plot_dim, ylim=plot_dim, zlim=plot_dim)
    plt.show()


def plot_statistics(open_filename):
    with open(open_filename, 'rb') as f:
        N, n_interacting, interaction_stats, interaction_idx_difference = pickle.load(f)

    s = 0.5
    fig, ax = plt.subplots(2 ,1 ,figsize=(8 ,6))
    ts = np.arange(len(interaction_stats[0]))
    ax[0].scatter(ts, interaction_stats[0], s=s, label='Interacting states')
    ax[0].scatter(ts, interaction_stats[1], s=s, label='Non-interacting states')

    # max_difference = np.max(np.nonzero(interaction_idx_difference)) + 1
    # ax[1].bar(np.arange(max_difference), interaction_idx_difference[:max_difference])

    ax[1].bar(np.arange(n_interacting), interaction_idx_difference)


    ax[0].set_xlabel(r'$t$', size=14)
    ax[0].set_ylabel('No. of interactions', size=14)
    ax[0].set_title(f'No. of nucleosomes = {N}', size=16)

    ax[1].set_xlabel('Index difference', size=14)
    ax[1].set_ylabel('No. of interactions', size=14)

    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

