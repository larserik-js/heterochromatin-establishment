
## External packages
from timeit import default_timer as timer

## Script runs the simulation
import run

##############################################################################
##############################################################################

## SET PARAMETERS
# Creates AND saves an animation
animate = False
# Saves pickle files and initial/final plots
save = True

# No. of nucleosomes
N = 150
# Equilibrium spring length
l0 = 1
# Half the spring constant
spring_strength = 100
# Noise
noise = 0.3*l0
# Time-step length
dt = 0.0001
# No. of time-steps
t_total = 100000

# Potential weights
U_spring_weight = 10
U_interaction_weight = 8000
U_pressure_weight = 0.0001
U_twist_weight = 1000
U_p_direction_weight = 100
#potential_weights = [U_spring_weight, U_interaction_weight, U_pressure_weight, U_twist_weight, U_p_direction_weight]
potential_weights = [U_spring_weight, U_interaction_weight, U_pressure_weight]

##############################################################################
##############################################################################

## RUN THE SCRIPT
if __name__ == '__main__':

    # Start the timer
    initial_time = timer()

    # Run the script
    run.run(N, spring_strength, l0, noise, potential_weights, dt, t_total, animate, save)

    # Print time elapsed
    print(f'Simulation finished at {timer()-initial_time:.2f} s')