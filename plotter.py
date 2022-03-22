from plot_class import Plots

# Plot parameters
plot_cell_division = False

plot_n_processes = 100
plot_U_pressure_weight = 0.17
plot_stats_interval = 100

plot_cenH_size = 8
plot_cenH_sizes = [6,8]
plot_cenH_init_idx = 16
plot_ATF1_idx = 30
plot_N = 40
plot_t_total = 30000
plot_noise = 0.5

initial_states = ['A', 'A_U', 'U', 'U_S', 'S']
plot_initial_state = initial_states[0]

dt = 0.02
plot_alpha_1 = 0.07
plot_alpha_2 = 0.1
plot_beta = 0.004
plot_seed = 8

plot_obj = Plots(plot_n_processes=plot_n_processes, plot_U_pressure_weight=plot_U_pressure_weight,
                 plot_stats_interval=plot_stats_interval, plot_cenH_size=plot_cenH_size,
                 plot_cenH_sizes=plot_cenH_sizes, plot_cenH_init_idx=plot_cenH_init_idx, plot_ATF1_idx=plot_ATF1_idx,
                 plot_cell_division=plot_cell_division, plot_N=plot_N, plot_t_total=plot_t_total, plot_noise=plot_noise,
                 plot_initial_state=plot_initial_state, plot_alpha_1=plot_alpha_1, plot_alpha_2=plot_alpha_2,
                 plot_beta=plot_beta, plot_seed=plot_seed)

# Plot final state
#plot_obj.plot_final_state()

#plot_obj.plot_interactions()

## Plot heatmap
#plot_obj.plot_heatmap()

# Plot correlation
#plot_obj.plot_correlation()

# Plot states
#plot_obj.plot_states()

# Plot states in time and space
plot_obj.plot_states_time_space()

# Plot end-to-end distance R
#plot_obj.plot_Rs()

#plot_obj.plot_RMS()

#plot_obj.plot_dynamics_time()

#plot_obj.plot_correlation_times()

# Plot end-to-end perpendicular times
#plot_obj.plot_end_to_end_times()

# Plot successful recruited conversions
#plot_obj.plot_successful_recruited_conversions()

#plot_obj.plot_fraction_ON_cells()

#plot_obj.plot_establishment_times_patches()

#plot_obj.plot_optimization()

#plot_obj.plot_res()