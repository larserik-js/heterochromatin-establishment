from plot_functions import plot_final_state, plot_interactions, plot_rg_vs_noise, plot_heatmap, plot_correlation,\
    plot_states, plot_Rs, plot_correlation_times

plot_noise = 0.5
plot_t_total = 16000000
plot_N = 40

constant = 1

dt = 0.02
plot_alpha_1 = 30 * dt * constant
plot_alpha_2 = 49 * dt * constant
plot_beta = 1 * dt * constant
plot_seed = 0

# # Plot end-to-end distance R
# plot_Rs(N=plot_N, t_total=plot_t_total, noise=plot_noise, alpha_1=plot_alpha_1, alpha_2=plot_alpha_2,
#                  beta=plot_beta, seed=plot_seed)
# plot_correlation_times(N=plot_N, t_total=plot_t_total, noise=plot_noise, alpha_1=plot_alpha_1, alpha_2=plot_alpha_2,
#             beta=plot_beta, seed=plot_seed)

## Plot final state
# plot_final_state(N=plot_N, t_total=plot_t_total, noise=plot_noise, alpha_1=plot_alpha_1, alpha_2=plot_alpha_2,
#                  beta=plot_beta, seed=plot_seed)


# #Plot statistics
# plot_interactions(N=plot_N, t_total=plot_t_total, noise=plot_noise, alpha_1=plot_alpha_1, alpha_2=plot_alpha_2,
#             beta=plot_beta, seed=plot_seed)
#
#
# ## Plot RG
# plot_rg_vs_noise(N=plot_N, t_total=plot_t_total, save=False)
#
# ## Plot heatmap
# plot_heatmap(plot_N, plot_t_total, save=False)
#
# plot_correlation(plot_noise, plot_t_total)

plot_states(N=plot_N, t_total=plot_t_total, noise=plot_noise, alpha_1=plot_alpha_1, alpha_2=plot_alpha_2,
            beta=plot_beta, seed=plot_seed)

