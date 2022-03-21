# Standard libraries
import os
import sys

# Own modules
import misc_functions

dt = 0.02
cenH_sizes = [6,7,8]
cenH_size = 8
cenH_init_idx = 16
initial_state = 'active'

# Bounds on rms_values: (1.677, 4.130)
rms_values = [2]
t_total = 30000

alpha_1 = 0.07
alpha_2 = 0.1
beta = 0.004
min_seed = 0


# Check if input RMS values are valid
if misc_functions.rms_vals_within_bounds(rms_values) == False:
    print('One or more RMS values outside of bounds. Enter valid values.')
    sys.exit()

# Execute commands
for rms in rms_values:
    print(f'rms: {rms:.2f}, cenH: {cenH_size}, '
            + f'cenH indices: {cenH_init_idx},...,{cenH_init_idx + cenH_size - 1}, alpha_1: {alpha_1:.4f}, '
            + f'alpha_2: {alpha_2:.4f}, beta: {beta:.4f}, seed: {min_seed}')

    os.system(f"python3 main.py --n_processes=25 --pool_size=25 --multiprocessing_parameter=seed "
              + f"--write_cenH_data=1 --animate=0 --allow_state_change=1 --t_total={t_total} --rms={rms} "
              + f"--initial_state={initial_state} --alpha_1={alpha_1} --alpha_2={alpha_2} --beta={beta} "
              + f"--min_seed={min_seed} --cenH_size={cenH_size} --cenH_init_idx={cenH_init_idx}")

