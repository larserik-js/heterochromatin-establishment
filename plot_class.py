import copy
import torch
import pickle
import matplotlib.pyplot as plt
from matplotlib import colors, patches
import numpy as np
from glob import glob
import seaborn as sns

# from mayavi import mlab
# from mayavi.mlab import *

from formatting import pathname, create_param_filename, create_plot_title

class Plots:
    def __init__(self, plot_cenH, plot_cell_division, plot_barriers, plot_N, plot_t_total, plot_noise,
                 plot_alpha_1, plot_alpha_2, plot_beta, plot_seed):

        self.cenH = plot_cenH
        self.cell_division = plot_cell_division
        self.barriers = plot_barriers
        self.N = plot_N
        self.t_total = plot_t_total
        self.noise = plot_noise
        self.alpha_1 = plot_alpha_1
        self.alpha_2 = plot_alpha_2
        self.beta = plot_beta
        self.seed = plot_seed

        self.state_colors = ['r', 'y', 'b']
        self.state_names = ['Silent', 'Unmodified', 'Active']

        self.pathname = pathname
        self.param_filename = create_param_filename(self.cenH, self.cell_division, self.barriers,
                                                    self.N, self.t_total, self.noise, self.alpha_1, self.alpha_2,
                                                    self.beta, self.seed)
        self.plot_title = create_plot_title(self.cenH, self.barriers, self.N, self.t_total, self.noise, self.alpha_1,
                                            self.alpha_2, self.beta, self.seed)
        r_system = self.N / 2
        self.plot_dim = (-0.5*r_system, 0.5*r_system)

    def create_full_filename(self, specific_filename, format):
        return pathname + specific_filename + self.param_filename + format

    def format_plot(self, ax, xlabel=',', ylabel=',', zlabel=None, legend_loc='best'):
        ax.set_xlabel(xlabel, size=12)
        ax.set_ylabel(ylabel, size=12)
        if zlabel is not None:
            ax.set_zlabel(zlabel, size=12)
        ax.set_title(self.plot_title, size=12)
        ax.legend(loc=legend_loc)
        plt.tight_layout()
        return None

    def plot_final_state(self):
        open_filename = self.create_full_filename('statistics/final_state/final_state_', '.pkl')

        with open(open_filename, 'rb') as f:
            x_plot, y_plot, z_plot, states, state_colors, state_names, plot_dim = pickle.load(f)

        # Center of mass
        com = np.array([x_plot.sum(), y_plot.sum(), z_plot.sum()]) / len(x_plot)

        # Create figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the different states
        for i in range(len(self.state_colors)):
            ax.scatter(x_plot[states == i], y_plot[states == i], z_plot[states == i], s=5, c=self.state_colors[i])

        # Plot chain line
        all_condition = torch.ones_like(torch.from_numpy(states), dtype=torch.bool)

        ax.plot(x_plot[all_condition], y_plot[all_condition], z_plot[all_condition],
                marker='o', ls='solid', markersize=1, c='k', lw=0.7)

        for i in range(len(self.state_colors)):
            ax.scatter([],[],c=self.state_colors[i],label=self.state_names[i])

        # Set plot dimensions
        ax.set(xlim=(com[0] + self.plot_dim[0], com[0] + self.plot_dim[1]),
               ylim=(com[1] + self.plot_dim[0], com[1] + self.plot_dim[1]),
               zlim=(com[2] + self.plot_dim[0], com[2] + self.plot_dim[1]))

        self.format_plot(ax, xlabel=r'$x$', ylabel=r'$y$', zlabel=r'$z$', legend_loc='upper left')
        plt.show()

    def plot_interactions(self):
        fig, ax = plt.subplots(2,1, figsize=(8,6))

        alpha = 0.5
        open_filename = self.create_full_filename('statistics/interactions/interactions_', '.pkl')

        # Finds all .pkl files for N = N
        files = glob(open_filename)

        if len(files) > 0:
            with open(files[0], 'rb') as f:
                N, noise, interaction_idx_difference, average_lifetimes = pickle.load(f)
                ax[0].bar(np.arange(N), average_lifetimes, alpha=alpha, label='Average lifetimes')
                ax[1].bar(np.arange(N), interaction_idx_difference, alpha=alpha, label='Interaction length')

        # Set plot title
        self.format_plot(ax[0], ylabel='Average lifetimes')
        ax[0].set_xticklabels([])
        ax[0].set_yscale('log')

        self.format_plot(ax[1], xlabel='Index difference', ylabel='Frequency')
        ax[1].set_yscale('log')

        plt.show()

    def plot_rg_vs_noise(self):
        polymer_types = ['classic', 'non-classic']
        # Plot
        fig, ax = plt.subplots(figsize=(8,6))

        for polymer_type in polymer_types:
            open_filename = self.create_full_filename('statistics/RG/', '.pkl')
            files = glob(open_filename)

            n_files = len(files)
            noise_list = np.empty(len(files))
            rg_list = np.empty(len(files))

            if n_files > 0:
                for i in range(n_files):
                    noise_list[i] = float(files[i].split('noise=')[1].split('.pkl')[0])

                    with open(files[i], 'rb') as f:
                        rg_list[i] = pickle.load(f)[0]

                ax.scatter(noise_list, rg_list, label=polymer_type)

        self.format_plot(ax, xlabel='Noise / ' + r'$l_0$', ylabel='Average radius of gyration')
        plt.show()

    def plot_heatmap(self):
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

        plt.tight_layout()
        plt.show()

    def plot_correlation(self):
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

                    ax.bar(shifts, correlation, alpha=alphas[i], label=labels[i])

        # Set plot title
        self.format_plot(ax, xlabel='Shift', ylabel='Average no. of interactions')
        ax.set_yscale('log')

        plt.show()

    def plot_states(self):
        open_filename = self.create_full_filename('statistics/states/states_', '.pkl')

        files = glob(open_filename)
        print(open_filename)

        n_files = len(files)

        if n_files == 0:
            print('No files to plot.')
            return

        fig,ax = plt.subplots(figsize=(8,6))

        for i in range(n_files):

            with open(files[i], 'rb') as f:
                state_statistics = pickle.load(f)[0]

            ts = torch.arange(len(state_statistics[0])) * (self.t_total / len(state_statistics[0]))

            lw = 0.2

            for j in range(len(self.state_names)):
                ax.plot(ts, state_statistics[j], lw=lw, c=self.state_colors[j], label=self.state_names[j])

        self.format_plot(ax, xlabel='Time-step', ylabel='No. of nucleosomes')
        plt.show()

    def plot_states_time_space(self):
        open_filename = self.create_full_filename('statistics/states_time_space/states_time_space_', '.pkl')

        files = glob(open_filename)
        print(open_filename)

        n_files = len(files)

        if n_files == 0:
            print('No files to plot.')
            return

        fig,ax = plt.subplots(figsize=(12,6))

        for i in range(n_files):

            with open(files[i], 'rb') as f:
                states_time_space = pickle.load(f)[0]

        print(states_time_space)

        labels = [patches.Patch(color=self.state_colors[i], label=self.state_names[i]) for i in range(len(self.state_colors))]
        print(labels)
        cmap = colors.ListedColormap(self.state_colors)
        self.format_plot(ax, xlabel='Time-steps / 2000', ylabel='Nucleosome no.')
        ax.imshow(states_time_space[::50].T, cmap=cmap)
        #ax.set_xlabel('Time-steps / 2000', size=12)
        #ax.set_ylabel('Nucleosome no.', size=12)
        plt.legend(handles=labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        plt.show()


    def plot_Rs(self):
        open_filename = self.create_full_filename('statistics/Rs/Rs_', '.pkl')

        files = glob(open_filename)

        n_files = len(files)

        if n_files == 0:
            print('No files to plot.')
            return

        fig,ax = plt.subplots(2, figsize=(8,6))
        s = 0.5

        self.format_plot(ax[0])

        for i in range(n_files):

            with open(files[i], 'rb') as f:
                Rs = pickle.load(f)[0]
                n_R = len(Rs)
                interval = int(self.t_total/n_R)
                ts = interval*np.arange(n_R)

            ax[0].scatter(ts, Rs, s=s, label='End-to-end distance')
            # When to start taking stats
            stats_idx = int(n_R/2)

            ax[1].hist(Rs[stats_idx:], bins=20, label='End-to-end distances')

            ax[0].set_xlabel(r'$t$')
            ax[1].set_xlabel(r'$R$')

        ax[1].legend(loc='best')
        fig.tight_layout()
        plt.show()

    def plot_correlation_times(self):
        open_filename = self.create_full_filename('statistics/correlation_times/correlation_times_', '.pkl')
        files = glob(open_filename)

        n_files = len(files)

        if n_files == 0:
            print('No files to plot.')
            return

        fig, ax = plt.subplots(figsize=(8, 6))

        for i in range(n_files):
            with open(files[i], 'rb') as f:
                correlation_times = pickle.load(f)[0]
                ax.bar(np.arange(self.N), correlation_times)
        self.format_plot(ax, xlabel='Nucleosome index', ylabel='Correlation time / ' + r'$t_{total}$')

        fig.tight_layout()
        plt.show()

    def plot_end_to_end_times(self):
        open_filename = pathname + f'statistics/end_to_end_perpendicular_times_N={self.N}_t_total={self.t_total}' \
                                   f'_noise={self.noise:.4f}' + '.txt'

        data_array = np.loadtxt(open_filename, skiprows=1, usecols=0, delimiter=',')

        mean = data_array.mean()
        std = data_array.std(ddof=1)

        fig, ax = plt.subplots()

        ax.text(400000, 200, f'Mean: {mean:.0f} +/- {std/np.sqrt(len(data_array)):.0f}', c='r')

        ax.set_xlabel('First time of perpendicular end-to-end vector')
        ax.set_ylabel('Frequency')
        ax.set_title(r'$N$' + f' = {self.N}, ' + r'$t_{total}$' + f' = {self.t_total}, '
                      + f'noise = {self.noise:.2f}' + r'$l_0$')
        ax.hist(data_array, bins=40)
        plt.show()
        return data_array



