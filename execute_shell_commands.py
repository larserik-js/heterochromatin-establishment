import os
import numpy as np

dt = 0.02
cenH_sizes = [6,8]
cenH_size = 6
cenH_init_idx = 16
initial_state = 'active'

pressure = 0.5
#pressure_list = np.arange(11) * 0.1
t_total = 100000

alpha_1 = 0.07
alpha_2 = 0.1
beta = 0.004
#beta = 0.1
min_seed = 3

print(f'pressure: {pressure:.2f}, cenH: {cenH_size}, cenH indices: {cenH_init_idx},...,{cenH_init_idx + cenH_size - 1}'
        + f', alpha_1: {alpha_1:.4f}, alpha_2: {alpha_2:.4f}, beta: {beta:.4f}, seed: {min_seed}')

os.system(f"python3 main.py --n_processes=1 --pool_size=1 --multiprocessing_parameter=seed --test_mode=0 "
          + f"--write_cenH_data=0 --animate=1 --allow_state_change=1 --t_total={t_total} --stats_t_interval=100 "
          + f"--U_pressure_weight={pressure} --initial_state={initial_state} --alpha_1={alpha_1} --alpha_2={alpha_2} "
          + f"--beta={beta} --min_seed={min_seed} --cenH_size={cenH_size} --cenH_init_idx={cenH_init_idx}")
