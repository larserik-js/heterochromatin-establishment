
## External packages
import torch
import numpy as np
from timeit import default_timer as timer
#from tqdm import tqdm
from torch.multiprocessing import Pool, cpu_count

## Script runs the simulation
import run

##############################################################################
##############################################################################

## SET PARAMETERS
# Multiprocessing
multi = False
if multi:
    torch.set_num_threads(1)

# Plots initial and final state, as well as statistics
# Nothing (except possibly an animation) is saved
test_mode = False

# Additionally generates and saves an animation
animate = False

# Random seed
seed = 4
seed_list = np.arange(30)

# No. of nucleosomes
N = 40
# Equilibrium spring length
l0 = 1
# Noise
noise_list = l0 * torch.linspace(0.05, 3.05, 31)
noise = 0.5*l0
# Time-step length
dt = 0.02
# No. of time-steps
t_total = 1000 # 200000

# Potential weights
U_two_interaction_weight = 50
U_pressure_weight = 1

## State parameters
# Allow states to change
allow_state_change = True

# Include cenH region
cenH = False

# Constants
constant = 1
constant_list = [0.1, 10]

# Recruited conversion probability
# Towards S
alpha_1 = 3.0 * dt * constant
# Towards A
alpha_2 = 4.9 * dt * constant
alpha_1_list = np.linspace(2.5, 3.5, 11) * dt
# Noisy conversion probability (given that a noisy conversion attempt is chosen)
beta = 0.2 * dt * constant

##############################################################################
##############################################################################

def curied_run(seed):
    #alpha_1 = alpha
    return run.run(N, l0, noise, dt, t_total, U_two_interaction_weight, U_pressure_weight, constant*alpha_1,
                   constant*alpha_2, constant*beta, seed,
                   test_mode=False, animate=animate, allow_state_change=allow_state_change, cenH=cenH, verbose=True)

## RUN THE SCRIPT
if __name__ == '__main__':

    # Get detailed error messages
    import torch
    torch.autograd.set_detect_anomaly(False)

    # Run the script
    total_time = 0
    if multi:
        # Start the timer
        print(f'Simulation (using multiprocessing) started.')
        initial_time = timer()

        # Create pool for multiprocessing
        pool = Pool(cpu_count())
        #pool = Pool(12)

        res = list(pool.map(curied_run, seed_list, chunksize=1))

        # Print time elapsed
        final_time = timer()-initial_time
        print(f'Simulation finished at {final_time:.2f} s')

    else:
        # Start the timer
        print(f'Simulation started. Noise = {noise:.2f}')
        initial_time = timer()

        # Run the simulation
        run.run(N, l0, noise, dt, t_total, U_two_interaction_weight, U_pressure_weight, alpha_1, alpha_2, beta, seed,
                test_mode, animate, allow_state_change, cenH, verbose=True)

        # Print time elapsed
        final_time = timer()-initial_time
        print(f'Simulation finished at {final_time:.2f} s')



