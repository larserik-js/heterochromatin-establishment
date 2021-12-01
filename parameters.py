import torch
import numpy as np

## SET PARAMETERS
# Multiprocessing
multi = True

# Plots initial and final state, as well as statistics
# Nothing (except possibly an animation) is saved
test_mode = False

# Additionally generates and saves an animation
animate = False

# Random seed
seed = 0
seed_list = np.arange(30)

# No. of nucleosomes
N = 40
# Equilibrium spring length
l0 = 1
# Noise
noise = 0.5*l0
noise_list = l0 * torch.linspace(0.05, 3.05, 31)
# Time-step length
dt = 0.02
# No. of time-steps
t_total = 16000000
# Time interval for taking statistics
stats_t_interval = 100

# Potential weights
U_two_interaction_weight = 50
U_pressure_weight = 1

## State parameters
# Allow states to change
allow_state_change = True

# Include cenH region
cenH = False
write_cenH_data = False

# Constants
constant = 1
constant_list = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]

# Recruited conversion probability
# Towards S
alpha_1 = 30 * dt * 0.1
alpha_1_list = np.linspace(25, 45, 21) * dt * 0.1
# Towards A
alpha_2 = 49 * dt * 0.1
# Noisy conversion probability
beta = 1.5 * dt * 0.1

pathname = '/home/lars/Documents/masters_thesis/'


##############################################################################
##############################################################################