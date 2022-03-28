import torch

# Write pressure and RMS values at the end of a simulation
def write_pressure_rms(output_dir, initial_state, cenH_size, cenH_init_idx, N, t_total, noise, alpha_1, alpha_2, beta,
                       seed, dist_vecs_to_com, U_pressure_weight):

    # Make filename
    write_name = output_dir + 'statistics/pressure_RMS_'
    write_name += f'init_state={initial_state}_cenH={cenH_size}_cenH_init_idx={cenH_init_idx}_N={N}_'\
                  f't_total={t_total}_noise={noise:.4f}_alpha_1={alpha_1:.5f}_alpha_2={alpha_2:.5f}_'\
                  f'beta={beta:.5f}_seed={seed}' + '.txt'

    # Append to the file
    shape_0 = dist_vecs_to_com.shape[0]
    # RMS for different time steps
    rms = torch.mean(torch.square(torch.norm(dist_vecs_to_com[int(shape_0/2):], dim=2)), dim=1)

    # This mean is a time average from t_half to the end of the simulation
    mean_rms = torch.mean(rms)
    line_str = f'{U_pressure_weight},{mean_rms:.4f}'
    data_file = open(write_name, 'a')
    data_file.write(line_str + '\n')
    data_file.close()
