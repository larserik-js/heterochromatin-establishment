## External packages
from timeit import default_timer as timer
import torch
from torch.multiprocessing import Pool, cpu_count
import numpy as np
from datetime import datetime

## Own scripts
import run
from formatting import pathname, create_directories
from parameters import get_parser_args

# Get all parameters
multi, test_mode, animate, seed, seed_list, N, l0, noise, noise_list, dt, t_total,stats_t_interval, \
U_two_interaction_weight, U_pressure_weight, allow_state_change, initial_state, initial_state_list, cell_division, cenH_size,\
cenH_init_idx, write_cenH_data, barriers, constant, constant_list, alpha_1, alpha_1_const, alpha_2, beta = get_parser_args()

alpha_1_list = np.linspace(25, 50, 101) * 0.02 * 0.1 * alpha_1_const
#alpha_1_list = np.linspace(25,50,2) * 0.02 * 0.1 * alpha_1_const
##############################################################################
##############################################################################

def curied_run(seed):
    return run.run(N, l0, noise, dt, t_total, U_two_interaction_weight, U_pressure_weight, constant*alpha_1,
                   constant*alpha_2, constant*beta, stats_t_interval, seed, False, animate, allow_state_change,
                   initial_state, cell_division, cenH_size, cenH_init_idx, write_cenH_data, barriers)

## RUN THE SCRIPT
if __name__ == '__main__':
    # Create necessary directories
    create_directories()

    # Get detailed error messages
    torch.autograd.set_detect_anomaly(False)

    # Run the script
    total_time = 0
    if multi:
        torch.set_num_threads(1)

        # Start the timer
        print(f'Simulation (using multiprocessing) started.')
        initial_time = timer()

        if not test_mode:

            # Creates the file for cenH statistics
            if (cenH_size > 0) and write_cenH_data:
                write_name = pathname + 'data/statistics/stable_silent_times_'
                write_name += f'pressure={U_pressure_weight:.2f}_init_state={initial_state}_cenH={cenH_size}_'\
                            + f'cenH_init_idx={cenH_init_idx}_N={N}_t_total={t_total}_noise={noise:.4f}_'\
                            + f'alpha_1={alpha_1:.5f}_alpha_2={alpha_2:.5f}_beta={beta:.5f}.txt'
                data_file = open(write_name, 'w')
                data_file.write('t,seed' + '\n')
                data_file.close()

            # # Write end-to-end vector data
            # write_name = f'{pathname}statistics/end_to_end_perpendicular_times_N={N}_t_total={t_total}_noise={noise:.4f}' \
            #               + f'_alpha_1={alpha_1:.5f}_alpha_2={alpha_2:.5f}_beta={beta:.5f}.txt'
            # data_file = open(write_name, 'w')
            # data_file.write('t,seed' + '\n')
            # data_file.close()

        # Create pool for multiprocessing
        #pool = Pool(cpu_count())
        pool = Pool(25)

        res = list(pool.map(curied_run, seed_list, chunksize=1))

        # Print time elapsed
        final_time = timer()-initial_time
        print(f'Simulation finished at {final_time:.2f} s, {datetime.now()}')

    else:
        # Start the timer
        print(f'Simulation started.')
        initial_time = timer()

        # Run the simulation
        run.run(N, l0, noise, dt, t_total, U_two_interaction_weight, U_pressure_weight, alpha_1, alpha_2, beta,
                stats_t_interval, seed, test_mode, animate, allow_state_change, initial_state, cell_division, cenH_size,
                cenH_init_idx, write_cenH_data, barriers)

        # Print time elapsed
        final_time = timer()-initial_time
        print(f'Simulation finished at {final_time:.2f} s, {datetime.now()}')

