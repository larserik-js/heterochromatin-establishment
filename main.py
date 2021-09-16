
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

## External packages
from timeit import default_timer as timer

## Own script with constants, functions, etc.
import functions

##############################################################################
##############################################################################

## SET PARAMETERS
# Creates AND saves an animation
animate = True
# Saves pickle files and initial/final plots
save = False

# No. of nucleosomes
N = 100
# Equilibrium spring length
l0 = 1
# Half the spring constant
spring_strength = 100
# Noise
noise = 0.1*l0
# Time-step length
dt = 0.0001
# No. of time-steps
t_total = 100

# Potential weights
U_spring_weight = 10
U_interaction_weight = 10
U_pressure_weight = 0.001
U_twist_weight = 1000
U_p_direction_weight = 100
potential_weights = [U_spring_weight, U_interaction_weight, U_pressure_weight, U_twist_weight, U_p_direction_weight]

##############################################################################
##############################################################################

## RUN THE SCRIPT
if __name__ == '__main__':

    # Start the timer
    initial_time = timer()

    # Run the script
    functions.run(N, spring_strength, l0, noise, potential_weights, dt, t_total, animate, save)

    # Print time elapsed
    print(f'Simulation finished at {timer()-initial_time:.2f} s')