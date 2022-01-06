import os
import numpy as np

dt = 0.02
cenH_sizes = [0,3]
initial_states = ['active', 'active_unmodified', 'unmodified', 'unmodified_silent', 'silent']
scaling_constant = 5

t_total = 2000000

for cenH_size in cenH_sizes:
    for initial_state in initial_states:
        alpha_1_const = 1
        alpha_2 = 0.1 * dt * 50
        beta = 0.004

        print(f'initial state: {initial_state}, cenH: {cenH_size}, alpha_2: {alpha_2:.4f}, beta: {beta:.4f}')
        os.system(f"python3 main.py --multi=1 --test_mode=0 --t_total={t_total} --stats_t_interval=100 --initial_state={initial_state} --alpha_1_const={alpha_1_const} --alpha_2={alpha_2} --beta={beta} --cenH_size={cenH_size}")

        alpha_1_const *= scaling_constant
        alpha_2 *= scaling_constant

        print(f'initial state: {initial_state}, cenH: {cenH_size}, alpha_2: {alpha_2:.4f}, beta: {beta:.4f}')
        os.system(f"python3 main.py --multi=1 --test_mode=0 --t_total={t_total} --stats_t_interval=100 --initial_state={initial_state} --alpha_1_const={alpha_1_const} --alpha_2={alpha_2} --beta={beta} --cenH_size={cenH_size}")

        beta *= scaling_constant

        print(f'initial state: {initial_state}, cenH: {cenH_size}, alpha_2: {alpha_2:.4f}, beta: {beta:.4f}')
        os.system(f"python3 main.py --multi=1 --test_mode=0 --t_total={t_total} --stats_t_interval=100 --initial_state={initial_state} --alpha_1_const={alpha_1_const} --alpha_2={alpha_2} --beta={beta} --cenH_size={cenH_size}")
