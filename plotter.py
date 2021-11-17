from plot_functions import plot_final_state, plot_interactions, plot_rg_vs_noise, plot_heatmap, plot_correlation, plot_states

plot_noise = 0.5
plot_t_total = 1000000
plot_N = 60

dt = 0.02
plot_alpha = 0.5
plot_alpha_1 = 34 * dt
plot_alpha_2 = 49 * dt
plot_beta = 1 * dt

## Plot final state
plot_final_state(N=plot_N, t_total=plot_t_total, noise=plot_noise, alpha_1=plot_alpha_1, alpha_2=plot_alpha_2, beta=plot_beta)


# Plot statistics
#plot_interactions(N=plot_N, noise=plot_noise,t_total=plot_t_total, save=False)
#
#
# ## Plot RG
# plot_rg_vs_noise(N=plot_N, t_total=plot_t_total, save=False)
#
# ## Plot heatmap
# plot_heatmap(plot_N, plot_t_total, save=False)
#
# plot_correlation(plot_noise, plot_t_total)

plot_states(N=plot_N, t_total=plot_t_total, noise=plot_noise, alpha_1=plot_alpha_1, alpha_2=plot_alpha_2, beta=plot_beta)