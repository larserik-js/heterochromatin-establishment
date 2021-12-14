from plot_class import Plots

# Plot parameters
plot_cenH = False
plot_barriers = False

plot_N = 40
plot_t_total = 16000000
plot_noise = 0.5

constant = 0.005
dt = 0.02
plot_alpha_1 = 41 * dt * constant
plot_alpha_2 = 49 * dt * constant
plot_beta = 15 * dt * constant
plot_seed = 0

plot_obj = Plots(plot_cenH=plot_cenH, plot_barriers=plot_barriers, plot_N=plot_N, plot_t_total=plot_t_total,
                 plot_noise=plot_noise, plot_alpha_1=plot_alpha_1, plot_alpha_2=plot_alpha_2, plot_beta=plot_beta,
                 plot_seed=plot_seed)
# Plot final state
#plot_obj.plot_final_state()

#plot_obj.plot_interactions()

#plot_obj.plot_rg_vs_noise()

## Plot heatmap
#plot_obj.plot_heatmap()

# Plot correlation
#plot_obj.plot_correlation()

# Plot states
plot_obj.plot_states()

# Plot end-to-end distance R
#plot_obj.plot_Rs()

#plot_obj.plot_correlation_times()

# Plot end-to-end perpendicular times
#plot_obj.plot_end_to_end_times()
