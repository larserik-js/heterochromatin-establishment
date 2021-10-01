
## External packages
import torch
torch.set_num_threads(1)
from timeit import default_timer as timer
from tqdm import tqdm
from torch.multiprocessing import Pool, cpu_count

## Script runs the simulation
import run

##############################################################################
##############################################################################

## SET PARAMETERS
# "Classic" polymer or our model
classic = False

# Multiprocessing
multi = False

# Plots initial and final state, as well as statistics
# Nothing is saved
test_mode = True

# Additionally generates and saves an animation
animate = False

# No. of nucleosomes
N = 100
# Equilibrium spring length
l0 = 1
# Half the spring constant
spring_strength = 100
# Noise
noise_list = l0 * torch.linspace(2.5, 7.5, 21)
noise = 5*l0
# Time-step length
dt = 0.001
# No. of time-steps
t_total = 1000000

# Potential weights
U_spring_weight = 0.1
if classic:
    U_interaction_weight = 0.1
else:
    U_interaction_weight = 500
U_pressure_weight = 0.0001*0
U_twist_weight = 1000
U_p_direction_weight = 100
#potential_weights = [U_spring_weight, U_interaction_weight, U_pressure_weight, U_twist_weight, U_p_direction_weight]
potential_weights = [U_spring_weight, U_interaction_weight, U_pressure_weight]

##############################################################################
##############################################################################

def curied_run(noise):
    return run.run(N, spring_strength, l0, noise, potential_weights, dt, t_total,
                   classic, test_mode=False, animate=False, verbose=True)

## RUN THE SCRIPT
if __name__ == '__main__':

    if multi:
        # Start the timer
        print(f'Simulation (using multiprocessing) started.')
        initial_time = timer()

        # Create pool for multiprocessing
        pool = Pool(cpu_count())
        #pool = Pool(12)

    # Run the script
    import torch
    torch.autograd.set_detect_anomaly(False)

    total_time = 0

    if multi:
        res = list(pool.map(curied_run, noise_list))

        # Print time elapsed
        final_time = timer()-initial_time
        print(f'Simulation finished at {final_time:.2f} s')

    else:
        # Start the timer
        print(f'Simulation started. Noise = {noise:.2f}')
        initial_time = timer()

        # Run the simulation
        run.run(N, spring_strength, l0, noise, potential_weights, dt, t_total, classic, test_mode, animate, verbose=True)

        # Print time elapsed
        final_time = timer()-initial_time
        print(f'Simulation finished at {final_time:.2f} s')
