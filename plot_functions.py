import copy
import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import seaborn as sns

# from mayavi import mlab
# from mayavi.mlab import *


def plot_final_state(N, noise, t_total, save):
    open_filename = f'/home/lars/Documents/masters_thesis/final_state/'\
                    + f'final_state_N={N}_t_total={t_total}_noise={noise:.2f}.pkl'

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

    if save:
        fig.savefig('/home/lars/Documents/masters_thesis/images/final_state')

    ax.legend(loc='upper left')
    ax.set_title(r'$N$' + f' = {len(x_plot)}, ' + r'$t_{total}$' + f' = {t_total}, noise = {noise}', size=18)
    ax.set(xlim=plot_dim, ylim=plot_dim, zlim=plot_dim)
    plt.tight_layout()
    plt.show()


def plot_statistics(N, noise, t_total, save):
    s = 0.5
    fig, ax = plt.subplots(2,1, figsize=(8,6))

    polymer_types = ['classic', 'non-classic']
    labels = ['Classic polymer', 'Non-classic polymer']
    alphas = [0.5,0.5]

    for i in range(len(polymer_types)):
        # Finds all .pkl files for N = N
        files = glob('/home/lars/Documents/masters_thesis/statistics/' + polymer_types[i]
                     + f'_statistics_N={N}_t_total={t_total}_noise={noise:.2f}.pkl')

        if len(files) > 0:
            with open(files[0], 'rb') as f:
                N, noise, interaction_idx_difference, average_lifetimes = pickle.load(f)
                ax[0].bar(np.arange(N), average_lifetimes, alpha=alphas[i], label=labels[i])
                ax[1].bar(np.arange(N), interaction_idx_difference, alpha=alphas[i], label=labels[i])

    ax[0].set_title(r'$N$' + f' = {N}, ' + r'$t_{total}$' + f' = {t_total/2:.0f}, noise = {noise}' + r'$l_0$', size=18)

    ax[0].set_xticklabels([])
    ax[0].set_ylabel('Average lifetimes', size=14)
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')

    ax[1].set_xlabel('Index difference', size=14)
    ax[1].set_ylabel('Frequency', size=14)

    if save:
        fig.savefig('/home/lars/Documents/masters_thesis/images/statistics')

    ax[0].legend(loc='best')
    plt.tight_layout()
    plt.show()

def plot_rg_vs_noise(N, t_total,save):
    polymer_types = ['classic', 'non-classic']
    # Plot
    fig, ax = plt.subplots(figsize=(8,6))
    ax.set_title(r'$N$' + f' = {N}, ' + r'$t_{total}$' + f' = {t_total/2:.0f}', size=18)
    ax.set_xlabel('Noise / ' + r'$l_0$', size=14)
    ax.set_ylabel('Average radius of gyration', size=14)

    for polymer_type in polymer_types:

        # Finds all .pkl files for N = N
        files = glob('/home/lars/Documents/masters_thesis/statistics/RG/' + polymer_type
                     + '_RG_N=' + str(N) + f'_t_total={t_total}_noise=' + '*.pkl')

        n_files = len(files)
        noise_list = np.empty(len(files))
        rg_list = np.empty(len(files))

        if n_files > 0:
            for i in range(n_files):
                noise_list[i] = float(files[i].split('noise=')[1].split('.pkl')[0])

                with open(files[i], 'rb') as f:
                    rg_list[i] = pickle.load(f)[0]

            ax.scatter(noise_list, rg_list, label=polymer_type)


    if save:
        fig.savefig('/home/lars/Documents/masters_thesis/images/rg_vs_noise')

    ax.legend(loc='best')
    plt.show()

def plot_heatmap(N, t_total, save):
    polymer_types = ['classic', 'non-classic']

    fig, ax = plt.subplots(2,2, sharex=True, sharey=True, figsize=(10,6))

    colorbar_labels = ['ln average lifetimes', 'ln no. of interactions']

    for j in range(len(polymer_types)):

        # Finds all .pkl files for N = N
        files = glob('/home/lars/Documents/masters_thesis/statistics/' + polymer_types[j]
                     + '_statistics_N=' + str(N) + f'_t_total={t_total}_noise=' + '*.pkl')

        # Sort filenames by noise level
        files = sorted(files)

        # This number might differ, depending on available files
        n_files = len(files)
        # Arrays for heatmap
        noise_list = np.empty(n_files)

        for k in range(n_files):
            with open(files[k], 'rb') as f:

                _, noise_list[k], _, interaction_idx_difference, average_lifetimes = pickle.load(f)

                if k == 0:
                    lifetimes_array = copy.deepcopy(average_lifetimes)
                    int_idx_diff_array = copy.deepcopy(interaction_idx_difference)
                else:
                    lifetimes_array = np.block([[lifetimes_array], [average_lifetimes]])
                    int_idx_diff_array = np.block([[int_idx_diff_array], [interaction_idx_difference]])

        if n_files > 0:
            # Create heatmaps
            sns.heatmap(np.log(lifetimes_array + 1e-2), ax=ax[0, j], cbar_kws={'label': colorbar_labels[0]})
            sns.heatmap(np.log(int_idx_diff_array + 1e-2), ax=ax[1, j], cbar_kws={'label': colorbar_labels[1]})

            # Set axis ticks and labels
            _, xtick_labels = plt.xticks()
            ytick_locs, _ = plt.yticks()
            ytick_labels = np.linspace(noise_list.min(), noise_list.max(), len(ytick_locs))
            ax[1,j].set_xticklabels(xtick_labels, rotation=0)
            ax[0,j].set_yticks(ytick_locs)
            ax[0,j].set_yticklabels(ytick_labels, rotation=0)
            ax[1,j].set_yticks(ytick_locs)
            ax[1,j].set_yticklabels(ytick_labels, rotation=0)

        ax[1,j].set_xlabel('Interaction length', size=14)

    # When sharey=True, this will invert the y axis of all four subplots
    ax[0,0].invert_yaxis()

    # Set titles
    fig.suptitle(r'$N$' + f' = {N}, ' + r'$t_{total}$' + f' = {t_total/2:.0f}', size=16)
    ax[0,0].set_title('Classic', size=16)
    ax[0,1].set_title('Non-classic', size=16)

    ax[0,0].set_ylabel('Noise / ' r'$l_0$', size=14)
    ax[1,0].set_ylabel('Noise / ' r'$l_0$', size=14)

    if save:
        fig.savefig('/home/lars/Documents/masters_thesis/images/heatmap')

    plt.tight_layout()
    plt.show()

def plot_correlation(noise, t_total):
    s = 0.5
    fig, ax = plt.subplots(figsize=(8,6))

    polymer_types = ['classic', 'non-classic']
    labels = ['Classic polymer', 'Non-classic polymer']
    alphas = [0.5,0.5]

    for i in range(len(polymer_types)):
        # Finds all .pkl files for N = N
        files = glob('/home/lars/Documents/masters_thesis/statistics/correlation/' + polymer_types[i]
                     + f'_correlation_N=100_t_total={t_total}_noise={noise:.2f}.pkl')

        if len(files) > 0:
            with open(files[0], 'rb') as f:
                correlation = pickle.load(f)[0]
                shifts = np.arange(1, len(correlation) + 1, 1)

                print(correlation)
                print(shifts)

                ax.bar(shifts, correlation, alpha=alphas[i], label=labels[i])

    ax.set_title(r'$N$' + ' = 100, ' + r'$t_{total}$' + f' = {t_total/2:.0f}, noise = {noise}' + r'$l_0$', size=18)

    ax.set_ylabel('Average no. of interactions', size=14)
    ax.set_yscale('log')
    ax.set_xlabel('Shift', size=14)

    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()



