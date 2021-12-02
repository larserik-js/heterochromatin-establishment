pathname = '/home/lars/Documents/masters_thesis/'

def create_param_filename(N, t_total, noise, alpha_1, alpha_2, beta, seed):
    return f'N={N}_t_total={t_total}_noise={noise:.4f}_alpha_1={alpha_1:.5f}_alpha_2={alpha_2:.5f}' \
           + f'_beta={beta:.5f}_seed={seed}'


def create_plot_title(N, t_total, noise, alpha_1, alpha_2, beta, seed):
    return r'$N$' + f' = {N}, ' + r'$t_{total}$' + f' = {t_total}, noise = {noise:.2f}' + r'$l_0$' + ', ' \
           + r'$\alpha_1$' + f' = {alpha_1:.5f}, ' + r'$\alpha_2$' + f' = {alpha_2:.5f}, ' + r'$\beta$' \
           + f' = {beta:.5f}'
