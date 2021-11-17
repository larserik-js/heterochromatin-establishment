
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
multi = True
if multi:
    torch.set_num_threads(1)

# Plots initial and final state, as well as statistics
# Nothing (except possibly an animation) is saved
test_mode = False

# Additionally generates and saves an animation
animate = False

# No. of nucleosomes
N = 60
# Equilibrium spring length
l0 = 1
# Noise
noise_list = l0 * torch.linspace(0.05, 3.05, 31)
noise = 0.5*l0
# Time-step length
dt = 0.02
# No. of time-steps
t_total = 25000000

# Potential weights
U_two_interaction_weight = 50
U_pressure_weight = 1

## State parameters
# Allow states to change
allow_state_change = True

# Recruited conversion probability
# Towards S
alpha_1 = 32 * dt
# Towards A
alpha_2 = 49 * dt
alpha_1_list = np.linspace(32, 33, 2) * dt
# Noisy conversion probability (given that a noisy conversion attempt is chosen)
beta = 1 * dt

##############################################################################
##############################################################################

def curied_run(alpha_1):
    #alpha_1 = alpha
    return run.run(N, l0, noise, dt, t_total, U_two_interaction_weight, U_pressure_weight, alpha_1, alpha_2, beta,
                   test_mode=False, animate=animate, allow_state_change=allow_state_change, verbose=True)

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

        res = list(pool.map(curied_run, alpha_1_list))

        # Print time elapsed
        final_time = timer()-initial_time
        print(f'Simulation finished at {final_time:.2f} s')

    else:
        # Start the timer
        print(f'Simulation started. Noise = {noise:.2f}')
        initial_time = timer()

        # Run the simulation
        run.run(N, l0, noise, dt, t_total, U_two_interaction_weight, U_pressure_weight, alpha_1, alpha_2, beta,
                test_mode, animate, allow_state_change, verbose=True)

        # Print time elapsed
        final_time = timer()-initial_time
        print(f'Simulation finished at {final_time:.2f} s')
