import os
import sys

import misc_functions


model = 'CMOL'

dt = 0.02
cenH_init_idx = 16
cenH_sizes = [6,7,8]
cenH_size = 8
ATF1_idx = None
initial_state = 'A'

# Bounds on rms_values: (1.677, 4.130)
rms_values = [2]
t_total = 10000

alpha_1 = 0.07
alpha_2 = 0.1
beta = 0.004
min_seed = 0

# Check if input RMS values are valid
if not misc_functions.rms_vals_within_bounds(rms_values):
    print('One or more RMS values outside of bounds. Enter valid values.')
    sys.exit()

# Execute commands
for rms in rms_values:
    print(f'rms: {rms:.2f}, cenH: {cenH_size}, alpha_1: {alpha_1:.4f}, '
          + f'alpha_2: {alpha_2:.4f}, beta: {beta:.4f}, seed: {min_seed}')

    os.system(f'python3 main.py --model={model} --n_processes=10 '
              + f'--pool_size=25 --multiprocessing_parameter=seed '
              + f'--write_cenH_data=0 --animate=1 --allow_state_change=1 '
              + f'--t_total={t_total} --rms={rms} '
              + f'--initial_state={initial_state} --alpha_1={alpha_1} '
              + f'--alpha_2={alpha_2} --beta={beta} --min_seed={min_seed} '
              + f'--cenH_size={cenH_size} --ATF1_idx={ATF1_idx}')

