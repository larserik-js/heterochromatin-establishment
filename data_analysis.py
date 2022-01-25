import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import seaborn as sns
from formatting import pathname

pressure = 0.5
initial_state = 'active'
cenH_sizes = [6,7,8]
cenH_init_idx = 16
N = 40
t_total = 1000000
noise = 0.5
alpha_1 = 0.3 / 5
alpha_2 = 0.5 / 5
beta = 0.004


def plot_silent_times(initial_state, cenH_sizes, cenH_init_idx, N, t_total, noise, alpha_1, alpha_2, beta):
    ## Histogram parameters
    n_bins = 20
    hist_range = np.linspace(0,t_total,n_bins)
    txt_string = ''

    for cenH_size in cenH_sizes:
        param_string = f'pressure={pressure:.2f}_init_state={initial_state}_cenH={cenH_size}_cenH_init_idx={cenH_init_idx}_' \
                       + f'N={N}_t_total={t_total}_noise={noise:.4f}_alpha_1={alpha_1:.5f}_' \
                       + f'alpha_2={alpha_2:.5f}_beta={beta:.5f}.txt'

        # First time where 90% of the polymer is silent
        silent_times = np.loadtxt(pathname + 'data/statistics/stable_silent_times_' + param_string,
                                  skiprows=1, usecols=0, delimiter=',')

        # No. of data points
        n_data = len(silent_times)

        #plt.hist(silent_times, bins=30, alpha=0.6, label=f'cenH size = {cenH_size}')
        #sns.displot(data=silent_times, kind='ecdf', x=hist_range)

        # Proportion of polymer that is not silent
        # Corresponds to the number of cells yielding fluorescence
        n_non_silent = np.empty(n_bins)

        # Plot cumulative distribution
        for i in range(n_bins):
            n_non_silent[i] = (silent_times > hist_range[i]).sum()

        # Add info to the text string
        txt_string += f'cenH size = {cenH_size}: Mean = {silent_times.mean():.3g} '\
                      + f'+/- {silent_times.std(ddof=1) / np.sqrt(n_data):.3g}' + '\n'

        plt.plot(hist_range, n_non_silent / len(silent_times), label=f'cenH size = {cenH_size}')

    # Create plot text
    plt.text(2.5e5, 0.05, txt_string, c='r', size=8)

    plt.xlabel(r'$t$', size=12)
    plt.ylabel('Proportion of non-silent polymers', size=12)
    plt.title(f'Heterochromatin establishment, pressure = {pressure:.2f}, ' + r'$\alpha_1$' + f' = {alpha_1}', size=14)
    plt.yscale('log')
    # Format y axis values to float
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))
    plt.legend(loc='best')
    plt.show()

    return None

plot_silent_times(initial_state, cenH_sizes, cenH_init_idx, N, t_total, noise, alpha_1, alpha_2, beta)