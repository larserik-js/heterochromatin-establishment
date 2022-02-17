## External packages
from timeit import default_timer as timer
import torch
from torch.multiprocessing import Pool
import numpy as np
from datetime import datetime
from functools import partial

## Own scripts
import run
from formatting import pathname, create_directories, create_param_string, edit_stable_silent_times_file

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

    # Start the timer
    print(f'Simulation started.')
    initial_time = timer()

    # Parameter arrays for parallel processes
    # Seed list
    if multiprocessing_parameter == 'seed':
        parameter_list = np.arange(n_processes) + min_seed
    # alpha_1 list
    elif multiprocessing_parameter == 'alpha_1':
        parameter_list = np.linspace(25, 50, n_processes) * 0.02 * 0.1 * alpha_1_const
    # U_pressure_weight list
    elif multiprocessing_parameter == 'U_pressure_weight':
        parameter_list = np.linspace(0,1,n_processes)
    # constant list
    elif multiprocessing_parameter == 'constant':
        parameter_list = constant * np.logspace(start=-2, stop=0, num=n_processes)
    else:
        raise AssertionError('Invalid multiprocessing parameter given.')

    # Multiprocessing
    if n_processes > 1:
        torch.set_num_threads(1)

        # Make the file for cenH statistics
        if not test_mode:
            if (cenH_size > 0) and write_cenH_data:
                # Write the file, and the first two lines
                line_str = f't_total={t_total}' + '\n' + 'silent_t,half_silent_t,n_patches,seed'
                edit_stable_silent_times_file(U_pressure_weight, initial_state, cenH_size, cenH_init_idx,
                                         cell_division, barriers, N, t_total, noise, alpha_1, alpha_2, beta, min_seed,
                                         line_str, action='w')

            # Write pressure and RMS values
            write_name = pathname + 'data/statistics/pressure_RMS_'
            write_name += f'init_state={initial_state}_cenH={cenH_size}_cenH_init_idx={cenH_init_idx}_N={N}_'\
                          f't_total={t_total}_noise={noise:.4f}_alpha_1={alpha_1:.5f}_alpha_2={alpha_2:.5f}_'\
                          f'beta={beta:.5f}_seed={min_seed}' + '.txt'

            # Append to the file
            line_str = 'U_pressure_weight,RMS'
            data_file = open(write_name, 'w')
            data_file.write(line_str + '\n')
            data_file.close()

        # Do not write cenH data in test mode
        else:
            pass

        # Create pool for multiprocessing
        pool = Pool(pool_size)

        res = list(pool.map(partial(curied_run, multiprocessing_parameter=multiprocessing_parameter, N=N, l0=l0,
                                    noise=noise, dt=dt, t_total=t_total,
                                    U_two_interaction_weight=U_two_interaction_weight,
                                    U_pressure_weight=U_pressure_weight, alpha_1=alpha_1, alpha_2=alpha_2, beta=beta,
                                    stats_t_interval=stats_t_interval, min_seed=min_seed, test_mode=test_mode,
                                    animate=animate, allow_state_change=allow_state_change, initial_state=initial_state,
                                    cell_division=cell_division, cenH_size=cenH_size, cenH_init_idx=cenH_init_idx,
                                    write_cenH_data=write_cenH_data, barriers=barriers),
                                    parameter_list, chunksize=1))

    # Run a single process without multiprocessing
    elif n_processes == 1:
        res = list(map(partial(curied_run, multiprocessing_parameter=multiprocessing_parameter, N=N, l0=l0,
                                    noise=noise, dt=dt, t_total=t_total,
                                    U_two_interaction_weight=U_two_interaction_weight,
                                    U_pressure_weight=U_pressure_weight, alpha_1=alpha_1, alpha_2=alpha_2, beta=beta,
                                    stats_t_interval=stats_t_interval, min_seed=min_seed, test_mode=test_mode,
                                    animate=animate, allow_state_change=allow_state_change, initial_state=initial_state,
                                    cell_division=cell_division, cenH_size=cenH_size, cenH_init_idx=cenH_init_idx,
                                    write_cenH_data=write_cenH_data, barriers=barriers),
                                    parameter_list))

        # Print time elapsed
        final_time = timer()-initial_time
        print(f'Simulation finished at {final_time:.2f} s, {datetime.now()}')

    else:
        raise AssertionError("n_processes set to 0. Choose a higher value.")

    return res



## RUN THE SCRIPT
if __name__ == '__main__':
    _ = main(N=N, l0=l0, noise=noise, dt=dt, t_total=t_total, U_two_interaction_weight=U_two_interaction_weight,
             U_pressure_weight=U_pressure_weight, alpha_1=alpha_1, alpha_2=alpha_2, beta=beta,
             stats_t_interval=stats_t_interval, min_seed=min_seed, test_mode=test_mode, animate=animate,
             allow_state_change=allow_state_change, initial_state=initial_state, cell_division=cell_division,
             cenH_size=cenH_size, cenH_init_idx=cenH_init_idx, write_cenH_data=write_cenH_data, barriers=barriers)


