import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import seaborn as sns
from formatting import pathname, create_directories
from scipy.optimize import curve_fit
from skopt import gp_minimize
import os
import main
from math import isfinite
from datetime import datetime
from estimation import Estimator

class Optimizer:
    def __init__(self, n_processes, pool_size, initial_state, cenH_init_idx, N, t_total, noise, U_pressure_weight,
                 alpha_2, beta, filename):
        self.n_processes = n_processes
        self.pool_size = pool_size
        self.initial_state = initial_state
        self.cenH_sizes = [6,8]
        self.cenH_init_idx = cenH_init_idx
        self.N = N
        self.t_total = t_total
        self.noise = noise
        self.U_pressure_weight = U_pressure_weight
        self.alpha_2 = alpha_2
        self.beta = beta
        self.filename = filename

        # Slope fraction(s) from the data
        # (cenH = 6) / (cenH = 8)
        self.slope_fraction_6_8 = 0.04891

    def write_best_param_data(self, alpha_1, tau_estimate_6, tau_estimate_6_error, tau_estimate_8,
                              tau_estimate_8_error, f_minimize_val):
        # Append to file
        data_file = open(self.filename, 'a')
        data_file.write(f'{self.U_pressure_weight},{alpha_1},{tau_estimate_6},{tau_estimate_6_error},'\
                            + f'{tau_estimate_8},{tau_estimate_8_error},{f_minimize_val}' + '\n')

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
    def f_minimize(self, alpha_1):
        alpha_1 = alpha_1[0]
        tau_estimates = []
        tau_estimates_errors = []

        for cenH_size in self.cenH_sizes:
            # Run the simulations
            silent_times_list = main.main(n_processes=self.n_processes, pool_size=self.pool_size, set_seed=False,
                                          t_total=self.t_total, U_pressure_weight=self.U_pressure_weight, alpha_1=alpha_1,
                                          cenH_size=cenH_size, test_mode=False, write_cenH_data=True)

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
        self.write_best_param_data(alpha_1, tau_estimates[0], tau_estimates_errors[0],
                                   tau_estimates[1], tau_estimates_errors[1], f_minimize_val)

        return f_minimize_val

    def optimize(self):
        # Minimize
        res = gp_minimize(self.f_minimize,  # The function to minimize

                          # The bounds on alpha_1
                          [(0.01, 0.2)],

                          # The acquisition function
                          acq_func="EI",

                          # The number of evaluations of f
                          n_calls=100,

                          # Number of evaluations of func with random points
                          # before approximating it with base_estimator.
                          n_initial_points=10,

                          # The noise level
                          noise="gaussian",

                          # The random seed
                          random_state=0)
        return res


U_pressure_weight_values = np.logspace(start=-5,stop=-3,num=3)
n_processes = 100
pool_size = 100
initial_state = 'active'
cenH_init_idx = 16
N = 40
t_total = 50000
noise = 0.5
alpha_2 = 0.1
beta = 0.004

def make_filename(U_pressure_weight, n_processes, initial_state, cenH_init_idx, N, t_total, noise, alpha_2, beta):
    return pathname + f'data/statistics/optimization/optimization_U_pressure_weight={U_pressure_weight:.2e}_'\
                    + f'n_processes={n_processes}_init_state={initial_state}_cenH_init_idx={cenH_init_idx}_N={N}_'\
                    + f't_total={t_total}_noise={noise:.4f}_alpha_2={alpha_2:.5f}_beta={beta:.5f}.txt'

def initialize_file(filename):
    data_file = open(filename, 'w')

    data_file.write('U_pressure_weight,alpha_1,tau_estimate(cenH=6),tau_estimate_error(cenH=6),' \
                    + 'tau_estimate(cenH=8),tau_estimate_error(cenH=8),f_minimize_val' + '\n')

    data_file.close()

if __name__ == '__main__':
    # Make necessary directories
    create_directories()

    # Iterate
    for U_pressure_weight in U_pressure_weight_values:
        # Make the .txt file for data
        filename = make_filename(U_pressure_weight, n_processes, initial_state,
                                 cenH_init_idx, N, t_total, noise, alpha_2, beta)
        initialize_file(filename)

        opt_obj = Optimizer(n_processes, pool_size, initial_state,
                            cenH_init_idx, N, t_total, noise, U_pressure_weight, alpha_2, beta, filename)

        res = opt_obj.optimize()
        print(res)

    print(f'Simulation finished {datetime.now()}')
