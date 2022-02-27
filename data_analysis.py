import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import seaborn as sns
from formatting import pathname
from scipy.optimize import curve_fit
from skopt import gp_minimize
import os
import main
from math import isfinite
from datetime import datetime
from estimation import Estimator

class Optimizer:
    def __init__(self, n_processes, pool_size, initial_state, cenH_init_idx, N, t_total, noise, alpha_1, alpha_2, beta):
        self.n_processes = n_processes
        self.pool_size = pool_size
        self.initial_state = initial_state
        self.cenH_sizes = [6,8]
        self.cenH_init_idx = cenH_init_idx
        self.N = N
        self.t_total = t_total
        self.noise = noise
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.beta = beta

        # self.n_bins = n_bins
        #
        # # Proportion of polymer that is not silent
        # # Corresponds to the number of cells yielding fluorescence
        # self.n_ON_cells = np.empty(self.n_bins)

        # Slope fraction(s) from the data
        # (cenH = 6) / (cenH = 8)
        self.slope_fraction_6_8 = 0.04891

    # def get_ON_cells_fractions(self):
    #     # Corresponds to the time of measurement in the (real) experiments
    #     self.measurement_times = np.linspace(0, self.t_total, self.n_bins+1)
    #
    #     # Find number of ON cells at a given measurement time
    #     for i in range(self.n_bins):
    #         self.n_ON_cells[i] = (self.silent_times > self.measurement_times[i]).sum()

    # # Fits a line to the data, returns the slope
    # @staticmethod
    # def fit_line(bin_centers, hist_values, yerr):
    #     popt, _ = curve_fit(lambda x, a, b: a*x + b, bin_centers, hist_values, sigma=yerr)
    #     return popt[0]


    def write_best_param_data(self, U_pressure_weight=None, tau_estimate_6=None, tau_estimate_6_error=None,
                              tau_estimate_8=None, tau_estiamte_8_error=None, f_minimize_val=None, action='w'):

        data_file = open(pathname + f'data/statistics/pressure_values_init_state={self.initial_state}_' \
                         + f'cenH_init_idx={self.cenH_init_idx}_N={self.N}_t_total={self.t_total}_' \
                         + f'noise={self.noise:.4f}_alpha_1={self.alpha_1:.5f}_alpha_2={self.alpha_2:.5f}_' \
                         + f'beta={self.beta:.5f}.txt', action)
        # Initialize file
        if action == 'w':
            data_file.write('U_pressure_weight,tau_estimate(cenH=6),tau_estimate_error(cenH=6),'\
                            + 'tau_estimate(cenH=8),tau_estimate_error(cenH=8),f_minimize_val' + '\n')
        # Append to file
        elif action == 'a':
            data_file.write(f'{U_pressure_weight},{tau_estimate_6},{tau_estimate_6_error},'\
                            + f'{tau_estimate_8},{tau_estiamte_8_error},{f_minimize_val}' + '\n')
        else:
            raise AssertionError('Invalid action given in function "write_best_param_data"!')

        data_file.close()

    def get_maxL_param(self, data):
        # The analytical result for the maximum likelihood value
        # of the parameter for an exponential function.
        # This is valid for a normalized distribution, which means that
        # the actual value only makes sense when we are using slope fractions
        #return len(self.n_ON_cells) / self.n_ON_cells.sum()
        estimator_obj = Estimator(T_MAX=self.t_total)
        _, tau_estimate, tau_estimate_error, _ = estimator_obj.estimate(data)

        return tau_estimate, tau_estimate_error


    # The function to be minimized
    def f_minimize(self, U_pressure_weight):
        U_pressure_weight = U_pressure_weight[0]
        tau_estimates = []
        tau_estimates_errors = []

        for cenH_size in self.cenH_sizes:
            # Run the simulations
            silent_times_list = main.main(n_processes=self.n_processes, pool_size=self.pool_size, t_total=self.t_total,
                                          U_pressure_weight=U_pressure_weight, cenH_size=cenH_size, test_mode=False,
                                          write_cenH_data=True)
            # # Remove 'None' values
            # silent_times_list = [time for time in silent_times_list if time is not None]
            # self.silent_times = np.array(silent_times_list)
            #
            # # Transform to ON cells fractions
            # self.get_ON_cells_fractions()

            tau_estimate, tau_estimate_error = self.get_maxL_param(silent_times_list)
            tau_estimates.append(tau_estimate)
            tau_estimates_errors.append(tau_estimate_error)

        # Divide to get actual slope fractions
        if tau_estimates[0] != None and tau_estimates != None:
            slope_cenH_6 = -1/tau_estimates[0]
            slope_cenH_8 = -1/tau_estimates[1]


            if slope_cenH_8 != 0:
                slope_fraction_data = slope_cenH_6 / slope_cenH_8
            else:
                raise AssertionError("Slope of value 0 found.")

            f_minimize_val = np.abs(self.slope_fraction_6_8 - slope_fraction_data)**2

        else:
            tau_estimates[0], tau_estimates[1] = 'NaN', 'NaN'
            tau_estimates_errors[0], tau_estimates_errors[1] = 'NaN', 'NaN'
            f_minimize_val = 99999

        # Write data continuously
        self.write_best_param_data(U_pressure_weight, tau_estimates[0], tau_estimates_errors[0],
                                   tau_estimates[1], tau_estimates_errors[1], f_minimize_val, action='a')

        return f_minimize_val

    def optimize(self):
        # Initialize file
        self.write_best_param_data(action='w')

        # Minimize
        res = gp_minimize(self.f_minimize,  # The function to minimize

                          # The bounds on U_pressure_weight
                          [(0.0, 1.0)],

                          # The acquisition function
                          acq_func="EI",

                          # The number of evaluations of f
                          n_calls=100,

                          # Number of evaluations of func with random points
                          # before approximating it with base_estimator.
                          n_initial_points=2,

                          # The noise level
                          noise="gaussian",

                          # The random seed
                          random_state=0)
        return res

    def plot_ON_cells_fractions(self):
        ## Histogram parameters
        n_bins = 20
        hist_range = np.linspace(0,t_total,n_bins)
        txt_string = ''

        # for cenH_size in cenH_sizes:
        #     param_string = f'pressure={pressure:.2f}_init_state={initial_state}_cenH={cenH_size}_cenH_init_idx={cenH_init_idx}_' \
        #                    + f'N={N}_t_total={t_total}_noise={noise:.4f}_alpha_1={alpha_1:.5f}_' \
        #                    + f'alpha_2={alpha_2:.5f}_beta={beta:.5f}.txt'
        #
        #     # First time where 90% of the polymer is silent
        #     silent_times = np.loadtxt(pathname + 'data/statistics/stable_silent_times_' + param_string,
        #                               skiprows=1, usecols=0, delimiter=',')
        #
        #     # No. of data points
        #     n_data = len(silent_times)
        #
        #     #plt.hist(silent_times, bins=30, alpha=0.6, label=f'cenH size = {cenH_size}')
        #     #sns.displot(data=silent_times, kind='ecdf', x=hist_range)
        #
        #     # Proportion of polymer that is not silent
        #     # Corresponds to the number of cells yielding fluorescence
        #     n_non_silent = np.empty(n_bins)
        #
        #     # Plot cumulative distribution
        #     for i in range(n_bins):
        #         n_non_silent[i] = (silent_times > hist_range[i]).sum()
        #
        #     # Add info to the text string
        #     txt_string += f'cenH size = {cenH_size}: Mean = {silent_times.mean():.3g} '\
        #                   + f'+/- {silent_times.std(ddof=1) / np.sqrt(n_data):.3g}' + '\n'
        #
        #     plt.plot(hist_range, n_non_silent / len(silent_times), label=f'cenH size = {cenH_size}')

        # Create plot text
        plt.text(2.5e5, 0.05, txt_string, c='r', size=8)

        plt.xlabel(r'$t$', size=12)
        plt.ylabel('Proportion of non-silent polymers', size=12)
        plt.title(f'Heterochromatin establishment, pressure = {pressure:.2f}, ' + r'$\alpha_1$' + f' = {alpha_1}', size=14)
        plt.yscale('log')
        # Format y axis values to float
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))
        plt.legend(loc='best')
        plt.show()

        return None

n_processes = 100
pool_size = 25
#pressure = 0.5
initial_state = 'active'
cenH_init_idx = 16
N = 40
t_total = 50000
noise = 0.5
alpha_1 = 0.07
alpha_2 = 0.1
beta = 0.004

opt_obj = Optimizer(n_processes, pool_size, initial_state, cenH_init_idx, N, t_total, noise, alpha_1, alpha_2, beta)
#opt_obj.plot_ON_cells_fractions()

res = opt_obj.optimize()
print(res)

print(f'Simulation finished {datetime.now()}')
