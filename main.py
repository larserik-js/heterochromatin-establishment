## External packages
from timeit import default_timer as timer
import torch
from torch.multiprocessing import Pool
import numpy as np
from datetime import datetime
from functools import partial

## Own scripts
import run
from formatting import pathname, create_directories

# Import all parameters
from parameters import n_processes, pool_size, multiprocessing_parameter, test_mode, animate, min_seed, N, l0, noise, dt,\
                       t_total, stats_t_interval, U_two_interaction_weight, U_pressure_weight, allow_state_change,\
                       initial_state, initial_state_list, cell_division, cenH_size, cenH_init_idx, write_cenH_data,\
                       barriers, constant, alpha_1, alpha_1_const, alpha_2, beta


##############################################################################
##############################################################################

def curied_run(x, multiprocessing_parameter, N, l0, noise, dt, t_total, U_two_interaction_weight, U_pressure_weight,
                alpha_1, alpha_2, beta, stats_t_interval, min_seed, test_mode, animate, allow_state_change,
                initial_state, cell_division, cenH_size, cenH_init_idx, write_cenH_data, barriers):

    if multiprocessing_parameter == 'seed':
        return run.run(N, l0, noise, dt, t_total, U_two_interaction_weight, U_pressure_weight, alpha_1,
                alpha_2, beta, stats_t_interval, x, test_mode, animate, allow_state_change,
                initial_state, cell_division, cenH_size, cenH_init_idx, write_cenH_data, barriers)

    elif multiprocessing_parameter == 'alpha_1':
        return run.run(N, l0, noise, dt, t_total, U_two_interaction_weight, U_pressure_weight, x,
                alpha_2, beta, stats_t_interval, min_seed, test_mode, animate, allow_state_change,
                initial_state, cell_division, cenH_size, cenH_init_idx, write_cenH_data, barriers)

    elif multiprocessing_parameter == 'U_pressure_weight':
        return run.run(N, l0, noise, dt, t_total, U_two_interaction_weight, x, alpha_1,
                alpha_2, beta, stats_t_interval, min_seed, test_mode, animate, allow_state_change,
                initial_state, cell_division, cenH_size, cenH_init_idx, write_cenH_data, barriers)
    else:
        raise AssertionError('Invalid multiprocessing_parameter given.')

def main(n_processes=n_processes, N=N, l0=l0, noise=noise, dt=dt, t_total=t_total,
             U_two_interaction_weight=U_two_interaction_weight,
             U_pressure_weight=U_pressure_weight, alpha_1=alpha_1, alpha_2=alpha_2, beta=beta,
             stats_t_interval=stats_t_interval, min_seed=min_seed, test_mode=test_mode, animate=animate,
             allow_state_change=allow_state_change, initial_state=initial_state, cell_division=cell_division,
             cenH_size=cenH_size, cenH_init_idx=cenH_init_idx, write_cenH_data=write_cenH_data, barriers=barriers):

    # Create necessary directories
    create_directories()

    # Get detailed error messages
    torch.autograd.set_detect_anomaly(False)

    # Run the script
    total_time = 0

    # Start the timer
    print(f'Simulation started.')
    initial_time = timer()

    if n_processes > 1:
        torch.set_num_threads(1)

        if not test_mode:

            # Creates the file for cenH statistics
            if (cenH_size > 0) and write_cenH_data:
                write_name = pathname + 'data/statistics/stable_silent_times/stable_silent_times_'
                write_name += f'pressure={U_pressure_weight:.2f}_init_state={initial_state}_cenH={cenH_size}_'\
                            + f'cenH_init_idx={cenH_init_idx}_N={N}_t_total={t_total}_noise={noise:.4f}_'\
                            + f'alpha_1={alpha_1:.5f}_alpha_2={alpha_2:.5f}_beta={beta:.5f}.txt'
                data_file = open(write_name, 'w')
                data_file.write(f't_total={t_total}' + '\n')
                data_file.write('t,seed' + '\n')
                data_file.close()

    # Parameter arrays for parallel processes
    # Seed list
    if multiprocessing_parameter == 'seed':
        parameter_list = np.arange(n_processes) + min_seed
    # alpha_1 list
    elif multiprocessing_parameter == 'alpha_1':
        parameter_list = np.linspace(25, 50, n_processes) * 0.02 * 0.1 * alpha_1_const
    # U_pressure_weight list
    elif multiprocessing_parameter == 'U_pressure_weight':
        parameter_list = np.arange(n_processes) * U_pressure_weight
    # constant list
    elif multiprocessing_parameter == 'constant':
        parameter_list = constant * np.logspace(start=-2, stop=0, num=n_processes)
    else:
        parameter_list = None
        raise AssertionError('Invalid multiprocessing parameter given.')

    # Create pool for multiprocessing
    pool = Pool(pool_size)

    # Run simulation(s)
    #res = list(pool.map(curied_run(), parameter_list, chunksize=1))
    res = list(pool.map(partial(curied_run,
                                multiprocessing_parameter=multiprocessing_parameter,
                                N=N, l0=l0, noise=noise, dt=dt, t_total=t_total,
                                U_two_interaction_weight=U_two_interaction_weight, U_pressure_weight=U_pressure_weight,
                                alpha_1=alpha_1, alpha_2=alpha_2, beta=beta, stats_t_interval=stats_t_interval,
                                min_seed=min_seed, test_mode=test_mode, animate=animate, allow_state_change=allow_state_change,
                                initial_state=initial_state, cell_division=cell_division, cenH_size=cenH_size,
                                cenH_init_idx=cenH_init_idx, write_cenH_data=write_cenH_data, barriers=barriers),
                                parameter_list, chunksize=1))

    # Print time elapsed
    final_time = timer()-initial_time
    print(f'Simulation finished at {final_time:.2f} s, {datetime.now()}')

    return res

## RUN THE SCRIPT
if __name__ == '__main__':
    _ = main(N=N, l0=l0, noise=noise, dt=dt, t_total=t_total, U_two_interaction_weight=U_two_interaction_weight,
             U_pressure_weight=U_pressure_weight, alpha_1=alpha_1, alpha_2=alpha_2, beta=beta,
             stats_t_interval=stats_t_interval, min_seed=min_seed, test_mode=test_mode, animate=animate,
             allow_state_change=allow_state_change, initial_state=initial_state, cell_division=cell_division,
             cenH_size=cenH_size, cenH_init_idx=cenH_init_idx, write_cenH_data=write_cenH_data, barriers=barriers)


