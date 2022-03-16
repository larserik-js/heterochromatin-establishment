import os

dt = 0.02
cenH_sizes = [6,7,8]
cenH_size = 8
cenH_init_idx = 16
initial_state = 'active'

# Bounds on rms_values: (1.677, inf)
rms_values = [1.677]
t_total = 10000

alpha_1 = 0.07
alpha_2 = 0.1
beta = 0.004
min_seed = 0

for rms in rms_values:
    print(f'rms: {rms:.2f}, cenH: {cenH_size}, '
            + f'cenH indices: {cenH_init_idx},...,{cenH_init_idx + cenH_size - 1}, alpha_1: {alpha_1:.4f}, '
            + f'alpha_2: {alpha_2:.4f}, beta: {beta:.4f}, seed: {min_seed}')

    os.system(f"python3 main.py --n_processes=25 --pool_size=25 --multiprocessing_parameter=seed --test_mode=0 "
              + f"--write_cenH_data=1 --animate=0 --allow_state_change=1 --t_total={t_total} --stats_t_interval=100 "
              + f"--rms={rms} --initial_state={initial_state} --alpha_1={alpha_1} "
              + f"--alpha_2={alpha_2} --beta={beta} --min_seed={min_seed} --cenH_size={cenH_size} "
              + f"--cenH_init_idx={cenH_init_idx}")

