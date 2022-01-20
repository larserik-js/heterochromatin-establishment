import os

dt = 0.02
cenH_sizes = [6,7,8]
cenH_size = 6
cenH_init_idx = 16
initial_state = 'active'

t_total = 1000

alpha_1 = 0.42 / 5
alpha_2 = 0.5 / 5
beta = 0.004
seed = 0

print(f'initial state: {initial_state}, cenH: {cenH_size}, cenH indices: {cenH_init_idx},...,{cenH_init_idx + cenH_size - 1}'
        + f', alpha_1: {alpha_1:.4f}, alpha_2: {alpha_2:.4f}, beta: {beta:.4f}, seed: {seed}')

os.system(f"python3 main.py --multi=0 --test_mode=1 --write_cenH_data=0 --animate=0 --t_total={t_total} --stats_t_interval=100" + " "
          + f"--initial_state={initial_state} --alpha_1={alpha_1} --alpha_2={alpha_2} --beta={beta}" + " "
          + f"--seed={seed} --cenH_size={cenH_size} --cenH_init_idx={cenH_init_idx}")
