import sys
from datetime import datetime

import numpy as np
from skopt import forest_minimize, dump

import main
from estimation import Estimator
from formatting import get_directories, make_output_directories
import misc_functions


class Optimizer:

    def __init__(self, model, run_on_cell, n_processes, pool_size,
                 initial_state, cenH_init_idx, N, t_total, noise, rms, alpha_2,
                 beta, filename):

        self.model = model
        self.run_on_cell = run_on_cell
        self.n_processes = n_processes
        self.pool_size = pool_size
        self.initial_state = initial_state
        self.cenH_sizes = [6,8]
        self.cenH_init_idx = cenH_init_idx
        self.N = N
        self.t_total = t_total
        self.noise = noise
        self.rms = rms
        self.alpha_2 = alpha_2
        self.beta = beta
        self.filename = filename

        # Slope fraction(s) from the data
        # (cenH = 6) / (cenH = 8)
        self.SLOPE_FRACTION_6_8 = 0.04891

    def write_best_param_data(self, alpha_1, tau_estimate_6,
                              tau_estimate_6_error, tau_estimate_8,
                              tau_estimate_8_error, f_minimize_val):
        # Append to file
        data_file = open(self.filename, 'a')
        data_file.write(f'{self.rms},{alpha_1},{tau_estimate_6},'
                        + f'{tau_estimate_6_error},{tau_estimate_8},'
                        + f'{tau_estimate_8_error},{f_minimize_val}' + '\n')

        data_file.close()

    def get_maxL_param(self, data):
        # The analytical result for the maximum likelihood value
        # of the parameter for an exponential function.
        # This is valid for a normalized distribution, which means that
        # the actual value only makes sense when we are using slope fractions
        #return len(self.n_ON_cells) / self.n_ON_cells.sum()
        estimator_obj = Estimator(t_max=self.t_total)
        _, tau_estimate, tau_estimate_error, _ = estimator_obj.estimate(data)

        return tau_estimate, tau_estimate_error

    # The function to be minimized
    def f_minimize(self, alpha_1):
        alpha_1 = alpha_1[0]
        tau_estimates = []
        tau_estimates_errors = []

        for cenH_size in self.cenH_sizes:
            # Run the simulations
            silent_times_list = main.main(
                model=self.model, run_on_cell=self.run_on_cell,
                n_processes=self.n_processes, pool_size=self.pool_size,
                t_total=self.t_total, rms=self.rms, alpha_1=alpha_1,
                cenH_size=cenH_size, set_seed=False, write_cenH_data=True)

            (tau_estimate,
             tau_estimate_error) = self.get_maxL_param(silent_times_list)
            tau_estimates.append(tau_estimate)
            tau_estimates_errors.append(tau_estimate_error)

        # Divide to get actual slope fractions
        if (tau_estimates[0] is not None) and (tau_estimates[1] is not None):
            slope_cenH_6 = -1/tau_estimates[0]
            slope_cenH_8 = -1/tau_estimates[1]

            # Calculate the fraction of the two slopes
            slope_fraction_data = slope_cenH_6 / slope_cenH_8

            # The value to be minimized
            f_minimize_val = np.abs(self.SLOPE_FRACTION_6_8
                                    - slope_fraction_data)**2

        else:
            tau_estimates[0], tau_estimates[1] = 'NaN', 'NaN'
            tau_estimates_errors[0], tau_estimates_errors[1] = 'NaN', 'NaN'
            f_minimize_val = 0.1 / alpha_1

        # Write data continuously
        self.write_best_param_data(
            alpha_1, tau_estimates[0], tau_estimates_errors[0],
            tau_estimates[1], tau_estimates_errors[1], f_minimize_val)

        return f_minimize_val

    def optimize(self):
        # Minimize
        res = forest_minimize(
            # The function to minimize
            self.f_minimize,

            # The bounds on alpha_1
            dimensions=[(0.01, 0.2)],

            # The acquisition function
            acq_func="EI",

            # The number of evaluations of f
            n_calls=5,

            # Number of evaluations of func with random points
            # before approximating it with base_estimator.
            n_initial_points=2,

            # The random seed
            random_state=0)

        return res


model = 'CMOL'
rms_values = [2]
n_processes = 25
pool_size = 25
initial_state = 'A'
cenH_init_idx = 16
N = 40
t_total = 20000
noise = 0.5
alpha_2 = 0.1
beta = 0.004
run_on_cell = False


def make_filename(output_dir, model, rms, n_processes, initial_state,
                  cenH_init_idx, N, t_total, noise, alpha_2, beta):

    return (output_dir + f'statistics/optimization/{model}_rms={rms:.3f}_'
            + f'n_processes={n_processes}_init_state={initial_state}_'
            + f'cenH_init_idx={cenH_init_idx}_N={N}_t_total={t_total}_'
            + f'noise={noise:.4f}_alpha_2={alpha_2:.5f}_beta={beta:.5f}.txt')


def initialize_file(filename):
    data_file = open(filename, 'w')

    data_file.write(
        'rms,alpha_1,tau_estimate(cenH=6),tau_estimate_error(cenH=6),'
        + 'tau_estimate(cenH=8),tau_estimate_error(cenH=8),f_minimize_val'
        + '\n')

    data_file.close()


def pickle_res(res, output_dir, rms, n_processes, initial_state, cenH_init_idx,
               N, t_total, noise, alpha_2, beta):

    filename = (output_dir + f'statistics/optimization/res_rms={rms:.3f}_'
                + f'n_processes={n_processes}_init_state={initial_state}_'
                + f'cenH_init_idx={cenH_init_idx}_N={N}_t_total={t_total}_'
                + f'noise={noise:.4f}_alpha_2={alpha_2:.5f}_beta={beta:.5f}'
                + '.pkl')

    # Write to pkl (using Skopt function)
    dump(res, filename, store_objective=False)


if __name__ == '__main__':
    # Make necessary directories
    project_dir, _, output_dir = get_directories(run_on_cell)
    make_output_directories(output_dir)

    # Iterate
    for rms in rms_values:
        # Check if input RMS values are valid
        if not misc_functions.rms_vals_within_bounds(rms_values):
            print('One or more RMS values outside of bounds. '
                  'Enter valid values.')
            sys.exit()

        # Make the .txt file for data
        filename = make_filename(
            output_dir, model, rms, n_processes, initial_state, cenH_init_idx,
            N, t_total, noise, alpha_2, beta)
        initialize_file(filename)

        opt_obj = Optimizer(
            model, run_on_cell, n_processes, pool_size, initial_state,
            cenH_init_idx, N, t_total, noise, rms, alpha_2, beta, filename)

        res = opt_obj.optimize()

        pickle_res(res, output_dir, rms, n_processes, initial_state,
                   cenH_init_idx, N, t_total, noise, alpha_2, beta)

        print(res)

    print(f'Simulation finished {datetime.now()}')