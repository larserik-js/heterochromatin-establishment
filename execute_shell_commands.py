import os
import sys

import numpy as np

import misc_functions


model = 'CMOL'
n_processes = 10

dt = 0.02
cenH_sizes = [0, 8]
cenH_size = 0
ATF1_idx = None
initial_state = 'S'
initial_states = ['A', 'S']
cell_division = 0

# Bounds on rms_values: (1.677, 4.130)
rms_values = [2.0, 4.0]
rms = 4.13
t_total = 10000000

alpha_1 = 0.07
#alpha_1_vals = list(np.arange(0.001, 0.01, 0.001)) + [0.2, 0.8, 0.9, 1.0]
alpha_2 = 0.1
beta = 0.004
min_seed = 0

# Check if input RMS values are valid
if not misc_functions.rms_vals_within_bounds(rms_values):
    print('One or more RMS values outside of bounds. Enter valid values.')
    sys.exit()

# Execute commands
print(f'rms: {rms:.2f}, cenH: {cenH_size}, alpha_1: {alpha_1:.4f}, '
      + f'alpha_2: {alpha_2:.4f}, beta: {beta:.4f}, seed: {min_seed}')

os.system(f'python3 main.py --model={model} '
          + f'--n_processes={n_processes} '
          + f'--pool_size=25 --multiprocessing_param=seed '
          + f'--write_cenH_data=0 --animate=0 --allow_state_change=1 '
          + f'--t_total={t_total} --rms={rms} '
          + f'--cell_division={cell_division} '
          + f'--initial_state={initial_state} --alpha_1={alpha_1} '
          + f'--alpha_2={alpha_2} --beta={beta} --min_seed={min_seed} '
          + f'--cenH_size={cenH_size} --ATF1_idx={ATF1_idx}')
