## External packages
from timeit import default_timer as timer
import torch
from torch.multiprocessing import Pool, cpu_count
import numpy as np
import argparse

## Own scripts
import run
# from parameters import multi, test_mode, animate, seed, seed_list, N, l0, noise, noise_list, dt, t_total,\
#                        stats_t_interval, U_two_interaction_weight, U_pressure_weight, allow_state_change, cell_division,\
#                        cenH, write_cenH_data, barriers, constant, constant_list, alpha_1, alpha_1_list, alpha_2, beta

from formatting import pathname

# Argparser from command line
parser = argparse.ArgumentParser()

# Multiprocessing
parser.add_argument('--multi',
                    type=bool,
                    default=False,
                    help='Use multiprocessing or not.')

# Plots initial and final state, as well as statistics
# Nothing (except possibly an animation) is saved
parser.add_argument('--test_mode',
                    type=bool,
                    default=True,
                    help='If false, saves data. Else, nothing is saved, except possibly an animation.')

# # Additionally generates and saves an animation
parser.add_argument('--animate',
                    type=bool,
                    default=False,
                    help='If true, creates an animation.')

# Random seed
parser.add_argument('--seed',
                    type=int,
                    default=0,
                    help='Random seed for Pytorch and Numpy.')

parser.add_argument('--seed_list',
                    default=np.arange(30),
                    help='Random seed list for Pytorch and Numpy.')

# No. of nucleosomes
parser.add_argument('--N',
                    type=int,
                    default=40,
                    help='The number of nucleosomes in the system.')

# Equilibrium spring length
parser.add_argument('--l0',
                    type=float,
                    default=1,
                    help='Equilibrium spring length.')

# Noise
parser.add_argument('--noise',
                    type=float,
                    default=0.5,
                    help='The dynamic noise level. Should be the product of some constant and the parameter l0.')

parser.add_argument('--noise_list',
                    default=torch.linspace(0.05, 3.05, 31),
                    help='List of dynamic noise levels. The values should be a product of some constant and the parameter l0.')

# Time-step size
parser.add_argument('--dt',
                    type=float,
                    default=0.02,
                    help='The time step size.')

# No. of time-steps
parser.add_argument('--t_total',
                    type=int,
                    default=1000,
                    help='The number of time-steps.')

# Time interval for taking statistics
parser.add_argument('--stats_t_interval',
                    type=int,
                    default=100,
                    help='The time-step interval at which to collect values for statistics.')

# Potential weights
parser.add_argument('--U_two_interaction_weight',
                    type=float,
                    default=50,
                    help='Scales the strength of the two-interaction potential.')

parser.add_argument('--U_pressure_weight',
                    type=float,
                    default=1,
                    help='Scales the strength of the external pressure potential.')

## State parameters
# Allow states to change
parser.add_argument('--allow_state_change',
                    type=bool,
                    default=True,
                    help='Enables the nucleosome states to change.')

# Allow cell division
parser.add_argument('--cell_division',
                    type=bool,
                    default=True,
                    help='Enables cell division.')

# Include cenH region
parser.add_argument('--cenH',
                    type=bool,
                    default=True,
                    help='Makes the cenH region active.')

parser.add_argument('--write_cenH_data',
                    type=bool,
                    default=False,
                    help='Collects data on the spreading of the silent state.')

# Include barriers
parser.add_argument('--barriers',
                    type=bool,
                    default=False,
                    help='Includes system barriers.')

# Constants
parser.add_argument('--constant',
                    type=float,
                    default=1,
                    help='The constant can be used to scale different parameters.')

parser.add_argument('--constant_list',
                    default=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1],
                    help='The constants can be used to scale different parameters.')

## Recruited conversion probabilities
# Towards S
parser.add_argument('--alpha_1',
                    type=float,
                    default=35*0.02*0.1,
                    help='The reaction rate for the recruited conversions of A to U and U to S.')

parser.add_argument('--alpha_1_list',
                    default=np.linspace(25, 49, 25) * 0.02 * 0.1,
                    help='The reaction rates for the recruited conversions of A to U and U to S.')

# Towards A
parser.add_argument('--alpha_2',
                    type=float,
                    default=49*0.02*0.1,
                    help='The reaction rate for the recruited conversions of S to U and U to A.')

## Noisy conversion probability
parser.add_argument('--beta',
                    type=float,
                    default=4*0.02*0.1,
                    help='The reaction rate for the noisy conversions of all states.')

# Gather all arguments in a namespace object
args = parser.parse_args()

# Extract all arguments
multi = args.multi
test_mode = args.test_mode
animate = args.animate
seed = args.seed
seed_list = args.seed_list
N = args.N
l0 = args.l0
noise = args.noise
noise_list = args.noise_list
dt = args.dt
t_total = args.t_total
stats_t_interval = args.stats_t_interval
U_two_interaction_weight = args.U_two_interaction_weight
U_pressure_weight = args.U_pressure_weight
allow_state_change = args.allow_state_change
cell_division = args.cell_division
cenH = args.cenH
write_cenH_data = args.write_cenH_data
barriers = args.barriers
constant = args.constant
constant_list = args.constant_list
alpha_1 = args.alpha_1
alpha_1_list = args.alpha_1_list
alpha_2 = args.alpha_2
beta = args.beta

##############################################################################
##############################################################################

def curied_run(seed):
    return run.run(N, l0, noise, dt, t_total, U_two_interaction_weight, U_pressure_weight, constant*alpha_1,
                   constant*alpha_2, constant*beta, stats_t_interval, seed, False, animate, allow_state_change,
                   cell_division, cenH, write_cenH_data, barriers)

## RUN THE SCRIPT
if __name__ == '__main__':

    # Get detailed error messages
    torch.autograd.set_detect_anomaly(False)

    # Run the script
    total_time = 0
    if multi:
        torch.set_num_threads(1)

        # Start the timer
        print(f'Simulation (using multiprocessing) started.')
        initial_time = timer()

        if not test_mode:

            # Creates the file for cenH statistics
            if cenH and write_cenH_data:
                write_name = pathname + 'statistics/stable_silent_times_' + 'cenH_'
                if barriers:
                    write_name += 'barriers_'
                write_name += f'N={N}_t_total={t_total}_noise={noise:.4f}'\
                                          + f'_alpha_1={alpha_1:.5f}_alpha_2={alpha_2:.5f}_beta={beta:.5f}.txt'
                data_file = open(write_name, 'w')
                data_file.write('t,seed' + '\n')
                data_file.close()

            # # Write end-to-end vector data
            # write_name = f'{pathname}statistics/end_to_end_perpendicular_times_N={N}_t_total={t_total}_noise={noise:.4f}' \
            #               + f'_alpha_1={alpha_1:.5f}_alpha_2={alpha_2:.5f}_beta={beta:.5f}.txt'
            # data_file = open(write_name, 'w')
            # data_file.write('t,seed' + '\n')
            # data_file.close()

        # Create pool for multiprocessing
        pool = Pool(cpu_count())

        res = list(pool.map(curied_run, seed_list, chunksize=1))

        # Print time elapsed
        final_time = timer()-initial_time
        print(f'Simulation finished at {final_time:.2f} s')

    else:
        # Start the timer
        print(f'Simulation started. Noise = {noise:.2f}')
        initial_time = timer()

        # Run the simulation
        run.run(N, l0, noise, dt, t_total, U_two_interaction_weight, U_pressure_weight, alpha_1, alpha_2, beta,
                stats_t_interval, seed, test_mode, animate, allow_state_change, cell_division, cenH, write_cenH_data,
                barriers)

        # Print time elapsed
        final_time = timer()-initial_time
        print(f'Simulation finished at {final_time:.2f} s')



