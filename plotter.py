from plot_class import Plots

plot_noise = 0.5
plot_t_total = 16000000
plot_N = 40

constant = 0.1

dt = 0.02
plot_alpha_1 = 30 * dt * constant
plot_alpha_2 = 49 * dt * constant
plot_beta = 1 * dt * constant
plot_seed = 0

plot_obj = Plots(plot_N=plot_N, plot_t_total=plot_t_total, plot_noise=plot_noise, plot_alpha_1=plot_alpha_1,
                 plot_alpha_2=plot_alpha_2, plot_beta=plot_beta, plot_seed=plot_seed)
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


