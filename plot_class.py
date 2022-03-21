import copy
import torch
import pickle
import matplotlib.pyplot as plt
from matplotlib import colors, patches
import numpy as np
from glob import glob
import seaborn as sns
import re
from numba import njit
import pandas as pd
from scipy import optimize
from skopt import plots, load

# from mayavi import mlab
# from mayavi.mlab import *

from formatting import get_project_folder, create_param_string, create_plot_title

class Plots:
    def __init__(self, plot_n_processes, plot_U_pressure_weight, plot_stats_interval, plot_cenH_size,
                 plot_cenH_sizes, plot_cenH_init_idx, plot_cell_division, plot_N, plot_t_total,
                 plot_noise, plot_initial_state, plot_alpha_1, plot_alpha_2, plot_beta, plot_seed):
    
        # Project folder
        self.pathname = get_project_folder()

        self.n_processes = plot_n_processes
        self.U_pressure_weight = plot_U_pressure_weight
        self.stats_interval = plot_stats_interval
        self.cenH_size = plot_cenH_size
        self.cenH_sizes = plot_cenH_sizes
        self.cenH_init_idx = plot_cenH_init_idx
        self.cell_division = plot_cell_division
        self.N = plot_N
        self.t_total = plot_t_total
        self.noise = plot_noise
        self.initial_state = plot_initial_state
        self.alpha_1 = plot_alpha_1
        self.alpha_2 = plot_alpha_2
        self.beta = plot_beta
        self.seed = plot_seed

        self.state_colors = ['r', 'y', 'b']
        self.state_names = ['Silent', 'Unmodified', 'Active']

        self.param_filename = create_param_string(self.U_pressure_weight, self.initial_state, self.cenH_size,
                                                  self.cenH_init_idx, self.cell_division, self.N, self.t_total,
                                                  self.noise, self.alpha_1, self.alpha_2, self.beta, self.seed)
        self.plot_title = create_plot_title(self.U_pressure_weight, self.cenH_size, self.cenH_init_idx, self.N,
                                            self.t_total, self.noise, self.alpha_1, self.alpha_2, self.beta, self.seed)
        r_system = self.N / 2
        self.plot_dim = (-0.5*r_system, 0.5*r_system)

    def create_full_filename(self, specific_filename, format):
        return self.pathname + specific_filename + self.param_filename + format

    def format_plot(self, ax, xlabel=',', ylabel=',', zlabel=None, legend_loc='best'):
        ax.set_xlabel(xlabel, size=12)
        ax.set_ylabel(ylabel, size=12)
        if zlabel is not None:
            ax.set_zlabel(zlabel, size=12)
        ax.set_title(self.plot_title, size=12)
        ax.legend(loc=legend_loc)
        plt.tight_layout()
        return None

    # State at the end of the simulation
    def plot_final_state(self):
        open_filename = self.create_full_filename('data/statistics/final_state/', '.pkl')

        with open(open_filename, 'rb') as f:
            x_plot, y_plot, z_plot, states = pickle.load(f)

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

    # Number of interactions and interaction lifetimes as a function of index difference
    def plot_interactions(self):
        fig, ax = plt.subplots(2,1, figsize=(8,6))

        TRANSPARENCY = 0.5
        open_filename = self.create_full_filename('data/statistics/interactions/', '.pkl')

        # Finds all .pkl files for N = N
        files = glob(open_filename)

        if len(files) > 0:
            with open(files[0], 'rb') as f:
                N, noise, interaction_idx_difference, average_lifetimes = pickle.load(f)
                ax[0].plot(np.arange(N), average_lifetimes, label='Average lifetimes')
                ax[1].bar(np.arange(N), interaction_idx_difference, alpha=TRANSPARENCY, label='Interaction length')

        # Set plot title
        self.format_plot(ax[0], ylabel='Average lifetimes')
        ax[0].set_xticklabels([])
        #ax[0].set_yscale('log')

        self.format_plot(ax[1], xlabel='Index difference', ylabel='Frequency')
        ax[1].set_yscale('log')

        plt.show()


    def plot_heatmap(self):
        polymer_types = ['classic', 'non-classic']

        fig, ax = plt.subplots(2,2, sharex=True, sharey=True, figsize=(10,6))

        colorbar_labels = ['ln average lifetimes', 'ln no. of interactions']

        for j in range(len(polymer_types)):

            # Finds all .pkl files for N = N
            files = glob('/home/lars/Documents/masters_thesis/statistics/' + polymer_types[j]
                         + '_N=' + str(N) + f'_t_total={t_total}_noise=' + '*.pkl')

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
                         + f'_N=100_t_total={t_total}_noise={noise:.2f}.pkl')

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
        open_filename = self.create_full_filename('statistics/states/', '.pkl')

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

            LINEWIDTH = 0.2

            for j in range(len(self.state_names)):
                ax.plot(ts, state_statistics[j], lw=LINEWIDTH, c=self.state_colors[j], label=self.state_names[j])

        self.format_plot(ax, xlabel='Time-step', ylabel='No. of nucleosomes')
        plt.show()

    def plot_states_time_space(self):
        open_filename = self.create_full_filename('data/statistics/states_time_space/', '.pkl')
        conversions_filename = self.create_full_filename('data/statistics/successful_conversions/', '.pkl')

        files = glob(open_filename)
        conversion_files = glob(conversions_filename)
        n_files = len(files)
        n_conversion_files = len(conversion_files)

        if n_files == 0 or n_conversion_files == 0:
            print('No files to plot.')
            return

        fig,ax = plt.subplots(figsize=(12,6))

        for i in range(n_files):
            with open(files[i], 'rb') as f:
                states_time_space = pickle.load(f)[0]
            with open(conversion_files[i], 'rb') as f_c:
                recruited_conversions, noisy_conversions = pickle.load(f_c)

        recruited_conversions = np.concatenate([recruited_conversions.sum(axis=1),
                                                np.array([recruited_conversions.sum()])
                                                ])

        noisy_conversions = np.concatenate([noisy_conversions, np.array([noisy_conversions.sum()])])

        df_array = np.block([[recruited_conversions],
                             [noisy_conversions]
                             ])

        # Show conversion data
        df = pd.DataFrame(df_array, index=['Recruited conversions / t', 'Noisy conversions / t'],
                                    columns=['S to U', 'U to A', 'A to U', 'U to S', 'Total'])
        print(df)
        # Plot
        INTERNAL_STATS_INTERVAL = 2

        labels = [patches.Patch(color=self.state_colors[i], label=self.state_names[i]) for i in range(len(self.state_colors))]
        cmap = colors.ListedColormap(self.state_colors)
        ax.imshow(states_time_space[::INTERNAL_STATS_INTERVAL].T, cmap=cmap)
        self.format_plot(ax, xlabel=f'Time-steps / {self.stats_interval * INTERNAL_STATS_INTERVAL}',
                         ylabel='Nucleosome no.')

        #ax.set_xlabel('Time-steps / 2000', size=12)
        #ax.set_ylabel('Nucleosome no.', size=12)
        plt.legend(handles=labels, bbox_to_anchor=(0.05, 2.3), loc=2, borderaxespad=0.)

        plt.show()

    def plot_Rs(self):
        open_filename = self.create_full_filename('statistics/Rs/', '.pkl')

        files = glob(open_filename)

        n_files = len(files)

        if n_files == 0:
            print('No files to plot.')
            return

        fig,ax = plt.subplots(2, figsize=(8,6))
        SCATTER_S = 0.5

        self.format_plot(ax[0])

        for i in range(n_files):

            with open(files[i], 'rb') as f:
                Rs = pickle.load(f)[0]
                n_R = len(Rs)
                interval = int(self.t_total/n_R)
                ts = interval*np.arange(n_R)

            ax[0].scatter(ts, Rs, s=SCATTER_S, label='End-to-end distance')
            # When to start taking stats
            stats_idx = int(n_R/2)

            ax[1].hist(Rs[stats_idx:], bins=20, label='End-to-end distances')

            ax[0].set_xlabel(r'$t$')
            ax[1].set_xlabel(r'$R$')

        ax[1].legend(loc='best')
        fig.tight_layout()
        plt.show()

    # Plots RMS as a function of pressure
    def plot_RMS(self):
        open_filename = self.create_full_filename('data/statistics/dist_vecs_to_com/', '.pkl')
        # Replace the pressure values with the wildcard * to include all pressure values
        open_filename = open_filename.replace(f'pressure={self.U_pressure_weight:.2f}', 'pressure=*')

        files = glob(open_filename)
        n_files = len(files)

        if n_files == 0:
            print('No files to plot.')
            return

        else:
            fig, ax = plt.subplots(figsize=(8,6))
            pressures, RMSs = [], []

            for i in range(n_files):
                # Get the pressure value
                search_result = re.search('pressure=(.*)_init_state', files[i])
                pressure_val = float(search_result.group(1))
                pressures.append(pressure_val)

                with open(files[i], 'rb') as f:
                    dist_vecs_to_com = pickle.load(f)[0]
                    t_idx, N = dist_vecs_to_com.shape[0], dist_vecs_to_com.shape[1]
                    RMSs.append(np.sqrt(np.square(dist_vecs_to_com).sum() / t_idx / N))

        ax.scatter(pressures, RMSs)
        ax.set_xlabel('Pressure', size=14)
        ax.set_ylabel('RMS', size=14)
        ax.set_title('RMS, ' + r'$t_{total}$' + f' = {self.t_total}', size=16)
        plt.show()


    # Vectors time correlation
    @staticmethod
    @njit
    def _calculate_correlations(correlations, dist_vecs_to_com, n_taus):
        i_tot, j_tot, k_tot = dist_vecs_to_com.shape[0], dist_vecs_to_com.shape[1], dist_vecs_to_com.shape[2]
        # Compute denominator

        denominator = 0

        for i in range(i_tot):
            for j in range(j_tot):
                for k in range(k_tot):
                    denominator += dist_vecs_to_com[i,j,k] ** 2

        denominator /= n_taus
        print('here')

        # Compute numerator
        for tau in range(n_taus):
            numerator = 0
            for i in range(n_taus - tau):
                for j in range(j_tot):
                    for k in range(k_tot):
                        numerator += dist_vecs_to_com[tau+i,j,k] * dist_vecs_to_com[i,j,k]

            numerator /= (n_taus - tau)
            correlations[tau] = numerator / denominator

        return correlations

    def plot_dynamics_time(self):
        open_filename = self.create_full_filename('data/statistics/dist_vecs_to_com/', '.pkl')

        files = glob(open_filename)
        n_files = len(files)

        if n_files == 0:
            print('No files to plot.')

        else:
            with open(files[0], 'rb') as f:
                dist_vecs_to_com = pickle.load(f)[0]

            # The time interval (no. of time steps) between each data point
            stats_t_interval = int(self.t_total / dist_vecs_to_com.shape[0])

            # The number of data points
            n_taus = dist_vecs_to_com.shape[0]
            correlations = np.empty(n_taus)

            denominator = (dist_vecs_to_com ** 2).sum() / n_taus

            for i in range(n_taus):
                numerator = (dist_vecs_to_com[:(n_taus - i)] * dist_vecs_to_com[i:]).sum() / (n_taus - i)
                correlations[i] = numerator / denominator

            #correlations = self._calculate_correlations(correlations, dist_vecs_to_com, n_taus)

            taus = np.arange(n_taus) * stats_t_interval

            fig, ax = plt.subplots(figsize=(8,6))

            ax.scatter(taus, correlations)
            ax.set(ylim=(-0.05, 1.05))

            ax.set_title('Dynamics time, ' + f'pressure = {self.U_pressure_weight:.2f}', size=16)
            ax.set_xlabel(r'$\tau$', size=14)
            ax.set_ylabel('Correlation', size=14)
            plt.show()

        return None


    def plot_correlation_times(self):
        open_filename = self.create_full_filename('data/statistics/correlation_times/', '.pkl')
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
        open_filename = self.pathname + f'statistics/end_to_end_perpendicular_times_N={self.N}_t_total={self.t_total}' \
                                 + f'_noise={self.noise:.4f}' + '.txt'

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

    def plot_successful_recruited_conversions(self):
        open_filename = self.create_full_filename('data/statistics/successful_recruited_conversions/', '.pkl')
        files = glob(open_filename)
        print(open_filename)

        fig, ax = plt.subplots(2,2, figsize=(12, 8))

        n_files = len(files)

        if n_files == 0:
            print('No files to plot.')
            return

        for i in range(n_files):
            with open(files[i], 'rb') as f:
                successful_recruited_conversions = pickle.load(f)[0]

        max_frequency = successful_recruited_conversions.max() + 1

        # S to U
        ax[0,0].bar(np.arange(self.N), successful_recruited_conversions[0])
        ax[0,0].set_title('S to U', size=14)
        ax[0,0].set_ylabel('Frequency', size=12)
        ax[0,0].set(ylim=(0,max_frequency))
        # U to A
        ax[0,1].bar(np.arange(self.N), successful_recruited_conversions[1])
        ax[0,1].set_title('U to A', size=14)
        ax[0,1].set(ylim=(0,max_frequency))
        # A to U
        ax[1,0].bar(np.arange(self.N), successful_recruited_conversions[2])
        ax[1,0].set_title('A to U', size=14)
        ax[1,0].set_xlabel('Index difference', size=12)
        ax[1,0].set_ylabel('Frequency', size=12)
        ax[1,0].set(ylim=(0,max_frequency))
        # U to S
        ax[1,1].bar(np.arange(self.N), successful_recruited_conversions[3])
        ax[1,1].set_title('U to S', size=14)
        ax[1,1].set_xlabel('Index difference', size=12)
        ax[1,1].set(ylim=(0,max_frequency))

        fig.suptitle('successful recruited conversions', size=16)

        fig.tight_layout()
        plt.show()

    def plot_fraction_ON_cells(self):
        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        txt_string = ''

        for cenH_size in self.cenH_sizes:
            param_string = f'pressure={self.U_pressure_weight:.2f}_init_state={self.initial_state}_cenH={cenH_size}_'\
                           + f'cenH_init_idx={self.cenH_init_idx}_N={self.N}_t_total={self.t_total}_'\
                           + f'noise={self.noise:.4f}_alpha_1={self.alpha_1:.5f}_alpha_2={self.alpha_2:.5f}'\
                           + f'_beta={self.beta:.5f}.txt'

            # First time where 90% of the polymer is silent
            try:
                silent_times = np.loadtxt(self.pathname + 'data/statistics/stable_silent_times/'
                                          + param_string, skiprows=2, usecols=0, delimiter=',')
            except:
                continue
            # No. of data points
            n_data = len(silent_times)

            # Keep only the finite values
            ts = silent_times[~np.isnan(silent_times)]
            k = len(ts)

            # Estimate
            tau_estimate = ts.mean() + (n_data/k-1) * self.t_total
            # The error
            second_derivative = k / tau_estimate ** 2 - 2 * ts.sum() / tau_estimate ** 3 - 2 * (n_data - k)\
                                * self.t_total / tau_estimate ** 2
            tau_estimate_error = np.sqrt(-1 / second_derivative)

            # Plot the fraction of 'ON' cells
            t_axis = np.linspace(0,5*self.t_total, 1000)
            ax.plot(t_axis, np.exp(-t_axis/tau_estimate), label=f'cenH={cenH_size}')

            # Add info to the text string
            txt_string += f'cenH size = {cenH_size}: tau estimate = {tau_estimate:.3g} +/- {tau_estimate_error:.3g}' + '\n'

        # Create plot text
        plt.text(2.5e5, 0.05, txt_string, c='r', size=8)

        ax.set_yscale('log')
        ax.legend(loc='best')
        plt.show()

    def plot_establishment_times_patches(self):
        # Plot
        fig, ax = plt.subplots(2,1, figsize=(7, 6))

        # The values to check for
        # Only values from data will eventually be used
        pressure_vals = np.arange(0,1.1,0.1)

        pressure_RMS_data = np.loadtxt(self.pathname + 'pressure_RMS.txt', delimiter=',')

        for cenH_size in self.cenH_sizes:
            # Values will only be appended to the lists if data exist
            RMS_list = []
            pressure_list = []
            est_time_list = []
            est_time_std_list = []
            n_patches_list = []
            n_patches_std_list = []

            for U_pressure_weight in pressure_vals:

                # Get corresponding RMS_val
                val_idx = np.argmin(np.abs(pressure_RMS_data[:,0] - U_pressure_weight))
                RMS = pressure_RMS_data[val_idx,1]

                param_string = f'pressure={U_pressure_weight:.2f}_init_state={self.initial_state}_cenH={cenH_size}_' \
                               + f'cenH_init_idx={self.cenH_init_idx}_N={self.N}_t_total={self.t_total}_' \
                               + f'noise={self.noise:.4f}_alpha_1={self.alpha_1:.5f}_alpha_2={self.alpha_2:.5f}' \
                               + f'_beta={self.beta:.5f}.txt'

                # Get data
                try:
                    data = np.loadtxt(self.pathname + 'data/statistics/stable_silent_times/'
                                      + param_string, skiprows=2, usecols=[0,1,2], delimiter=',')
                except:
                    continue

                silent_times = data[:,0]
                half_silent_times = data[:, 1]
                n_patches = data[:,2]

                # Picks out the finite data
                not_NaNs = ~np.isnan(silent_times)
                n_data = not_NaNs.sum() / len(not_NaNs)

                # ESTABLISHMENT TIMES
                # Keep only finite values
                establishment_times = silent_times[not_NaNs] - half_silent_times[not_NaNs]

                est_time_list.append(establishment_times.mean())
                est_time_std_list.append(establishment_times.std(ddof=1) / np.sqrt(n_data))

                # N_PATCHES
                n_patches = n_patches[not_NaNs]
                n_patches_list.append(n_patches.mean())
                n_patches_std_list.append(n_patches.std(ddof=1) / np.sqrt(n_data))

                #
                pressure_list.append(U_pressure_weight)
                RMS_list.append(RMS)

            # Plot
            ax[0].errorbar(RMS_list, est_time_list, yerr=est_time_std_list, fmt='-o', label=f'cenH = {cenH_size}')
            ax[1].errorbar(RMS_list, n_patches_list, yerr=n_patches_std_list, fmt='-o', label=f'cenH = {cenH_size}')

        ax[0].set_ylabel('Time from 50% - 90% silent', size=12)
        ax[1].set_xlabel('RMS', size=12)
        ax[1].set_ylabel('Mean number of silent islands', size=12)

        ax[0].legend(loc='best')
        ax[1].legend(loc='best')
        fig.tight_layout()
        plt.show()

    # # Plots the minimum f_minimize_vals as a function of pressure values, from a .txt document
    # def plot_optimization(self):
    #     filename = self.pathname + f'data/statistics/optimization_init_state={self.initial_state}_'\
    #                       + f'cenH_init_idx={self.cenH_init_idx}_N={self.N}_t_total={self.t_total}_'\
    #                       + f'noise={self.noise:.4f}_alpha_2={self.alpha_2:.5f}_beta={self.beta:.5f}.txt'
    #
    #     # Read the data into a Pandas DF
    #     df = pd.read_csv(filename, usecols=['U_pressure_weight', 'f_minimize_val'])
    #     # Group the minimum f_minimize_vals by U_pressure_weight
    #     df = df.groupby('U_pressure_weight')['f_minimize_val'].min()
    #     # Transform to Numpy arrays
    #     pressure_vals = df.index.to_numpy()
    #     f_min_vals = df.to_numpy()
    #
    #     fig, ax = plt.subplots(figsize=(8,6))
    #     ax.scatter(pressure_vals, f_min_vals)
    #     ax.set_xlabel('U_pressure_weight', size=14)
    #     ax.set_ylabel('f_minimize_val', size=14)
    #     plt.show()

    def plot_optimization(self):
        # Finds all .txt files with different pressure values
        filenames = glob(self.pathname + 'data/statistics/optimization/U_pressure_weight=*'\
                         + f'n_processes={self.n_processes}_init_state={self.initial_state}_'\
                         + f'cenH_init_idx={self.cenH_init_idx}_N={self.N}_t_total={self.t_total}_'\
                         + f'noise={self.noise:.4f}_alpha_2={self.alpha_2:.5f}_beta={self.beta:.5f}.txt')

        # Sort filenames by pressure values
        filenames = sorted(filenames)

        # This number might differ, depending on available files
        n_files = len(filenames)

        pressure_vals = np.empty(n_files)
        f_min_vals = np.empty(n_files)

        fig, ax = plt.subplots(figsize=(8,6))

        for k in range(n_files):
            with open(filenames[k], 'rb') as f:
                data = np.loadtxt(f, delimiter=',', skiprows=1, usecols=[0,-1])
                pressure_vals[k] = data[0,0]
                f_min_vals[k] = data[:,1].min()

        ax.scatter(pressure_vals, f_min_vals)
        plt.show()

    def plot_res(self):
        open_filename = self.pathname + f'data/statistics/optimization/'\
                         + f'res_U_pressure_weight={self.U_pressure_weight:.2e}_'\
                         + f'n_processes={self.n_processes}_init_state={self.initial_state}_'\
                         + f'cenH_init_idx={self.cenH_init_idx}_N={self.N}_t_total={self.t_total}_'\
                         + f'noise={self.noise:.4f}_alpha_2={self.alpha_2:.5f}_beta={self.beta:.5f}.pkl'

        # Load file (using Skopt function)
        res = load(open_filename)

        print('Best found =', res.x)

        plots.plot_gaussian_process(res)

        loss_func = lambda x0: res['models'][-1].predict(np.asarray(x0).reshape(-1, 1))

        # min_fun_res = optimize.minimize_scalar(loss_func, bounds=(0, 1), method='bounded').x
        min_fun_res = optimize.brute(loss_func, [(0, 1)], Ns=300, disp=True)

        true_x0 = res['space'].inverse_transform(min_fun_res.reshape(1, 1))
        print('SURROGATE MINIMUM =', true_x0)
        plt.show()