from plot_functions import plot_final_state, plot_statistics, plot_rg_vs_noise, plot_heatmap

plot_noise = 5.0
plot_t_total = 1000000
plot_N = 100

## Plot final state
plot_final_state(noise=plot_noise, t_total=plot_t_total,save=False)


## Plot statistics
plot_statistics(noise=plot_noise,t_total=plot_t_total, save=False)


## Plot RG
plot_rg_vs_noise(t_total=plot_t_total,N=plot_N,save=False)

## Plot heatmap
plot_heatmap(plot_t_total, plot_N, save=False)