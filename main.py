## External packages
from timeit import default_timer as timer
import torch
from torch.multiprocessing import Pool, cpu_count

## Own scripts
import run
from parameters import multi, test_mode, animate, seed, seed_list, N, l0, noise, noise_list, dt, t_total,\
                       stats_t_interval, U_two_interaction_weight, U_pressure_weight, allow_state_change, cenH,\
                       write_cenH_data, constant, constant_list, alpha_1, alpha_1_list, alpha_2, beta, pathname

##############################################################################
##############################################################################

def curied_run(alpha_1):
    return run.run(N, l0, noise, dt, t_total, U_two_interaction_weight, U_pressure_weight, constant*alpha_1,
                   constant*alpha_2, constant*beta, stats_t_interval, seed, test_mode=False, animate=animate,
                   allow_state_change=allow_state_change, cenH=cenH, write_cenH_data=write_cenH_data, verbose=True)

## RUN THE SCRIPT
if __name__ == '__main__':

    # Get detailed error messages
    torch.autograd.set_detect_anomaly(False)

    # Run the script
    total_time = 0
    if multi:
        torch.set_num_threads(1)

        # Start the timer
        print(f'Simulation (using multiprocessing) started.')
        initial_time = timer()

        # Creates the file for cenH statistics
        if cenH:
            write_cenH_data = True
            data_file = open(pathname + f'statistics/stable_silent_times_N={N}_t_total={t_total}_noise={noise:.4f}'\
                                      + f'_alpha_1={alpha_1:.5f}_alpha_2={alpha_2:.5f}_beta={beta:.5f}.txt', 'w')
            data_file.write('t,seed' + '\n')
            data_file.close()

        # Create pool for multiprocessing
        pool = Pool(cpu_count())

        res = list(pool.map(curied_run, alpha_1_list, chunksize=1))

        # Print time elapsed
        final_time = timer()-initial_time
        print(f'Simulation finished at {final_time:.2f} s')

    else:
        # Start the timer
        print(f'Simulation started. Noise = {noise:.2f}')
        initial_time = timer()

        # Run the simulation
        run.run(N, l0, noise, dt, t_total, U_two_interaction_weight, U_pressure_weight, alpha_1, alpha_2, beta,
                stats_t_interval, seed, test_mode, animate, allow_state_change, cenH, write_cenH_data,
                verbose=True)

        # Print time elapsed
        final_time = timer()-initial_time
        print(f'Simulation finished at {final_time:.2f} s')



