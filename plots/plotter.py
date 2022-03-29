from plot_class import Plots

# Plot parameters
plot_cell_division = False

plot_model = 'CMOL'

plot_n_processes = 25
# Bounds on rms_values: (1.677, 4.130)
plot_rms = 2
plot_stats_interval = 100

plot_cenH_size = 8
plot_cenH_sizes = [6,8]
plot_cenH_init_idx = 16
plot_ATF1_idx = None
plot_N = 40
plot_t_total = 20000
plot_noise = 0.5

initial_states = ['A', 'A_U', 'U', 'U_S', 'S']
plot_initial_state = initial_states[0]

dt = 0.02
plot_alpha_1 = 0.07
plot_alpha_2 = 0.1
plot_beta = 0.004
plot_seed = 8

plot_obj = Plots(model=plot_model, n_processes=plot_n_processes, rms=plot_rms, stats_interval=plot_stats_interval,
                 cenH_size=plot_cenH_size, cenH_sizes=plot_cenH_sizes, cenH_init_idx=plot_cenH_init_idx,
                 ATF1_idx=plot_ATF1_idx, cell_division=plot_cell_division, N=plot_N, t_total=plot_t_total,
                 noise=plot_noise, initial_state=plot_initial_state, alpha_1=plot_alpha_1, alpha_2=plot_alpha_2,
                 beta=plot_beta, seed=plot_seed)

# Plot final state
#plot_obj.final_state()

#plot_obj.interactions()

## Plot heatmap
#plot_obj.heatmap()

# Plot correlation
#plot_obj.correlation()

# Plot states
#plot_obj.states()

# Plot states in time and space
#plot_obj.states_time_space()

# Plot end-to-end distance R
#plot_obj.Rs()

#plot_obj.RMS()

#plot_obj.dynamics_time()

#plot_obj.correlation_times()

# Plot end-to-end perpendicular times
#plot_obj.end_to_end_times()

# Plot successful recruited conversions
#plot_obj.successful_recruited_conversions()

#plot_obj.fraction_ON_cells()

#plot_obj.establishment_times_patches()

#plot_obj.optimization()

plot_obj.res()