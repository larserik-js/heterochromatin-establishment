import torch
import numpy as np
import argparse

# ## SET PARAMETERS
# # Multiprocessing
# multi = True
#
# # Plots initial and final state, as well as statistics
# # Nothing (except possibly an animation) is saved
# test_mode = True
#
# # Additionally generates and saves an animation
# animate = False
#
# # Random seed
# seed = 0
# seed_list = np.arange(20)
#
# # No. of nucleosomes
# N = 40
# # Equilibrium spring length
# l0 = 1
# # Noise
# noise = 0.5*l0
# noise_list = l0 * torch.linspace(0.05, 3.05, 31)
# # Time-step length
# dt = 0.02
# # No. of time-steps
# t_total = 20000000
# # Time interval for taking statistics
# stats_t_interval = 1000
#
# # Potential weights
# U_two_interaction_weight = 50
# U_pressure_weight = 1
#
# ## State parameters
# # Allow states to change
# allow_state_change = True
#
# # Allow cell division
# cell_division = True
#
# # Include cenH region
# cenH = True
# write_cenH_data = False
#
# # Include barriers
# barriers = False
#
# # Constants
# constant = 1
# constant_list = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
#
# # Recruited conversion probability
# # Towards S
# alpha_1 = 37 * dt * 0.1
# alpha_1_list = np.linspace(25, 49, 25) * dt * 0.1
# # Towards A
# alpha_2 = 49 * dt * 0.1
# # Noisy conversion probability
# beta = 4 * dt * 0.1

# # Argparser from command line
# parser = argparse.ArgumentParser()
#
# # Multiprocessing
# parser.add_argument('--multi',
#                     default=False,
#                     help='Use multiprocessing or not.')
#
# # Plots initial and final state, as well as statistics
# # Nothing (except possibly an animation) is saved
# parser.add_argument('--test_mode',
#                     default=True,
#                     help='If false, saves data. Else, nothing is saved, except possibly an animation.')
#
# # # Additionally generates and saves an animation
# parser.add_argument('--animate',
#                     default=False,
#                     help='If true, creates an animation.')
#
# # Random seed
# parser.add_argument('--seed',
#                     default=0,
#                     help='Random seed for Pytorch and Numpy.')
#
# parser.add_argument('--seed_list',
#                     default=np.arange(20),
#                     help='Random seed list for Pytorch and Numpy.')
#
# # No. of nucleosomes
# parser.add_argument('--N',
#                     default=40,
#                     help='The number of nucleosomes in the system.')
#
# # Equilibrium spring length
# parser.add_argument('--l0',
#                     default=1,
#                     help='Equilibrium spring length.')
#
# # Noise
# parser.add_argument('--noise',
#                     default=0.5,
#                     help='The dynamic noise level. Should be the product of some constant and the parameter l0.')
#
# parser.add_argument('--noise_list',
#                     default=torch.linspace(0.05, 3.05, 31),
#                     help='List of dynamic noise levels. The values should be a product of some constant and the parameter l0.')
#
# # Time-step size
# parser.add_argument('--dt',
#                     default=0.02,
#                     help='The time step size.')
#
# # No. of time-steps
# parser.add_argument('--t_total',
#                     default=1000,
#                     help='The number of time-steps.')
#
# # Time interval for taking statistics
# parser.add_argument('--stats_t_interval',
#                     default=100,
#                     help='The time-step interval at which to collect values for statistics.')
#
# # Potential weights
# parser.add_argument('--U_two_interaction_weight',
#                     default=50,
#                     help='Scales the strength of the two-interaction potential.')
#
# parser.add_argument('--U_pressure_weight',
#                     default=1,
#                     help='Scales the strength of the external pressure potential.')
#
# ## State parameters
# # Allow states to change
# parser.add_argument('--allow_state_change',
#                     default=True,
#                     help='Enables the nucleosome states to change.')
#
# # Allow cell division
# parser.add_argument('--cell_division',
#                     default=True,
#                     help='Enables cell division.')
#
# # Include cenH region
# parser.add_argument('--cenH',
#                     default=False,
#                     help='Makes the cenH region active.')
#
# parser.add_argument('--write_cenH_data',
#                     default=False,
#                     help='Collects data on the spreading of the silent state.')
#
# # Include barriers
# parser.add_argument('--barriers',
#                     default=False,
#                     help='Includes system barriers.')
#
# # Constants
# parser.add_argument('--constant',
#                     default=1,
#                     help='The constant can be used to scale different parameters.')
#
# parser.add_argument('--constant_list',
#                     default=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1],
#                     help='The constants can be used to scale different parameters.')
#
# ## Recruited conversion probabilities
# # Towards S
# parser.add_argument('--alpha_1',
#                     default=35*0.02*0.1,
#                     help='The reaction rate for the recruited conversions of A to U and U to S.')
#
# parser.add_argument('--alpha1_list',
#                     default=np.linspace(25, 49, 25) * 0.02 * 0.1,
#                     help='The reaction rates for the recruited conversions of A to U and U to S.')
#
# # Towards A
# parser.add_argument('--alpha_2',
#                     default=49*0.02*0.1,
#                     help='The reaction rate for the recruited conversions of S to U and U to A.')
#
# ## Noisy conversion probability
# parser.add_argument('--beta',
#                     default=4*0.02*0.1,
#                     help='The reaction rate for the noisy conversions of all states.')
#
# # Gather all arguments in a namespace object
# args = parser.parse_args()

##############################################################################
##############################################################################