import os
import numpy as np

dt = 0.02
cenH_sizes = [0,3]

scaling_constant = 5

n_alpha_1_vals = 2

for cenH_size in cenH_sizes:
    alpha_1_values = 0.1 * dt * np.linspace(26, 50, n_alpha_1_vals)
    alpha_2 = 0.1 * dt * 50
    beta = 0.004

    for alpha_1 in alpha_1_values:
        print(f'cenH: {cenH_size}, alpha_1: {alpha_1:.4f}, alpha_2: {alpha_2:.4f}, beta: {beta:.4f}')
        os.system(f"python3 main.py --multi=1 --test_mode=0 --t_total=1000 --stats_t_interval=10 --alpha_1={alpha_1} --alpha_2={alpha_2} --beta={beta} --cenH_size={cenH_size}")

    alpha_2 *= scaling_constant
    alpha_1_values = 0.1 * dt * np.linspace(25, 50, n_alpha_1_vals) * scaling_constant

    for alpha_1 in alpha_1_values:
        print(f'cenH: {cenH_size}, alpha_1: {alpha_1:.4f}, alpha_2: {alpha_2:.4f}, beta: {beta:.4f}')
        os.system(f"python3 main.py --multi=1 --test_mode=0 --t_total=1000 --stats_t_interval=10 --alpha_1={alpha_1} --alpha_2={alpha_2} --beta={beta} --cenH_size={cenH_size}")

    beta *= scaling_constant

    for alpha_1 in alpha_1_values:
        print(f'cenH: {cenH_size}, alpha_1: {alpha_1:.4f}, alpha_2: {alpha_2:.4f}, beta: {beta:.4f}')
        os.system(f"python3 main.py --multi=1 --test_mode=0 --t_total=1000 --stats_t_interval=10 --alpha_1={alpha_1} --alpha_2={alpha_2} --beta={beta} --cenH_size={cenH_size}")