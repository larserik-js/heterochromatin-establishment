import os
import numpy as np

dt = 0.02
cenH_sizes = [1,2]
cenH_size = 0
cenH_init_idx = 16
initial_state = 'active'

pressure = 0.1
#pressure_list = np.arange(11) * 0.1
t_total = 1000000

alpha_1 = 0.35 / 5
alpha_2 = 0.5 / 5
beta = 0.004
seed = 0

print(f'pressure: {pressure:.2f}, cenH: {cenH_size}, cenH indices: {cenH_init_idx},...,{cenH_init_idx + cenH_size - 1}'
        + f', alpha_1: {alpha_1:.4f}, alpha_2: {alpha_2:.4f}, beta: {beta:.4f}, seed: {seed}')

os.system(f"python3 main.py --multi=1 --test_mode=0 --write_cenH_data=0 --animate=0 --allow_state_change=0 "
          + f"--t_total={t_total} --stats_t_interval=100 --U_pressure_weight={pressure} "
          + f"--initial_state={initial_state} --alpha_1={alpha_1} --alpha_2={alpha_2} --beta={beta} "
          + f"--seed={seed} --cenH_size={cenH_size} --cenH_init_idx={cenH_init_idx}")
