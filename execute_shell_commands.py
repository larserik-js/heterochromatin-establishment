import os
# from formatting import pathname
# from parameters import multi, test_mode, animate, seed, seed_list, N, l0, noise, noise_list, dt, t_total,\
#                        stats_t_interval, U_two_interaction_weight, U_pressure_weight, allow_state_change, cenH,\
#                        write_cenH_data, barriers, constant, constant_list, alpha_1, alpha_1_list, alpha_2, beta


os.system(f"python3 main.py --multi=True --test_mode=False --t_total=20000000 --stats_t_interval=1000")
