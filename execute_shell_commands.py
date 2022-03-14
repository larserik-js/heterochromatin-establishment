import os
import numpy as np

dt = 0.02
cenH_sizes = [6,7,8]
cenH_size = 8
cenH_init_idx = 16
initial_state = 'active'

pressure = 0.01
#pressure_list = np.arange(11) * 0.1
pressure_list = [0.1]
t_total = 10000

alpha_1 = 0.07
alpha_2 = 0.1
beta = 0.004
#beta = 0.1
min_seed = 0

for pressure in pressure_list:
    print(f'pressure: {pressure:.2f}, cenH: {cenH_size}, cenH indices: {cenH_init_idx},...,{cenH_init_idx + cenH_size - 1}'
            + f', alpha_1: {alpha_1:.4f}, alpha_2: {alpha_2:.4f}, beta: {beta:.4f}, seed: {min_seed}')

    os.system(f"python3 main.py --n_processes=25 --pool_size=25 --multiprocessing_parameter=seed --test_mode=0 "
              + f"--write_cenH_data=1 --animate=0 --allow_state_change=1 --t_total={t_total} --stats_t_interval=100 "
              + f"--U_pressure_weight={pressure} --initial_state={initial_state} --alpha_1={alpha_1} --alpha_2={alpha_2} "
              + f"--beta={beta} --min_seed={min_seed} --cenH_size={cenH_size} --cenH_init_idx={cenH_init_idx}")

