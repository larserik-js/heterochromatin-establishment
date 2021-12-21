from plot_class import Plots

# Plot parameters
plot_cell_division = False
plot_barriers = False

plot_stats_interval = 10

plot_cenH_size = 0
plot_N = 40
plot_t_total = 1000
plot_noise = 0.5
plot_initial_state='active'

constant = 0.1
dt = 0.02
plot_alpha_1 = 25 * dt * constant
plot_alpha_2 = 50 * dt * constant
plot_beta = 2 * dt * constant
plot_seed = 0

plot_obj = Plots(plot_stats_interval=plot_stats_interval, plot_cenH_size=plot_cenH_size, plot_cell_division=plot_cell_division,
                 plot_barriers=plot_barriers, plot_N=plot_N, plot_t_total=plot_t_total, plot_noise=plot_noise,
                 plot_initial_state=plot_initial_state, plot_alpha_1=plot_alpha_1, plot_alpha_2=plot_alpha_2,
                 plot_beta=plot_beta, plot_seed=plot_seed)

# Plot final state
#plot_obj.plot_final_state()

#plot_obj.plot_interactions()

#plot_obj.plot_rg_vs_noise()

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

#plot_obj.plot_correlation_times()

# Plot end-to-end perpendicular times
#plot_obj.plot_end_to_end_times()
