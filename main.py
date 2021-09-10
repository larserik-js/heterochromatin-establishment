
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

## External packages
import numpy as np
from timeit import default_timer as timer

## Own script with constants, functions, etc.
import torch

import functions

##############################################################################
##############################################################################

## SET PARAMETERS
animate=False

# No. of nucleosomes
N = 50
# Equilibrium spring length
l0 = 1
# Half the spring constant
spring_strength = 100
# Noise
noise = 0.1*l0
# Time-step length
dt = 0.0001
# No. of time-steps
t_total = 2000

# Potential weights
U_spring_weight = 10
U_interaction_weight = 50
U_pressure_weight = 0.001
potential_weights = [U_spring_weight, U_interaction_weight, U_pressure_weight]

# Gather all parameters
parameters = [N, spring_strength, l0, noise, potential_weights, dt, t_total, animate]

##############################################################################
##############################################################################

## RUN THE SCRIPT
if __name__ == '__main__':

    # Start the timer
    initial_time = timer()

    # Run the script
    functions.run(parameters)

    # Print time elapsed
    print(f'Simulation finished at {timer()-initial_time:.2f} s')