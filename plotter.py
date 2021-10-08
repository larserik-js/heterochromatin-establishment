from plot_functions import plot_final_state, plot_statistics, plot_rg_vs_noise, plot_heatmap, plot_correlation

plot_noise = 2
plot_t_total = 20000
plot_N = 100

## Plot final state
plot_final_state(N=plot_N, noise=plot_noise, t_total=plot_t_total,save=False)


## Plot statistics
plot_statistics(N=plot_N, noise=plot_noise,t_total=plot_t_total, save=False)


## Plot RG
plot_rg_vs_noise(N=plot_N, t_total=plot_t_total, save=False)

## Plot heatmap
plot_heatmap(plot_N, plot_t_total, save=False)

plot_correlation(plot_noise, plot_t_total)