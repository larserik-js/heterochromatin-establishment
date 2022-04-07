import numpy as np
import matplotlib.pyplot as plt
import mpmath
import tqdm as tqdm
from scipy import optimize


r = np.random


class Estimator:

    def __init__(self, tau_true=2, t_max=5, n_samples=100, n_runs=10):
        self.tau_true = tau_true
        self.t_max = t_max
        # Number of samples
        self.n_samples = n_samples
        # Number of runs
        self.n_runs = n_runs

    def estimate(self, data=None, print_stats=False):
        # If no input data, samples data randomly
        if data is None:
            # All sampled data
            data = r.exponential(scale=self.tau_true, size=self.n_samples)
            are_below_t_max = (data < self.t_max)

        # Experimental data given as argument
        else:
            data = np.array(data)
            are_below_t_max = np.array([val is not None for val in data])
            pass

        # No data below t_max
        if not np.any(are_below_t_max):
            return None, None, None, None
        else:
            # Data sampled from continuous distribution
            ts = data[are_below_t_max]
            k = len(ts)
            N = len(data)

            # Terms for the estimate of tau
            term_1 = ts.mean()
            term_2 = (N / k - 1) * self.t_max

            tau_estimate = term_1 + term_2

            # The error
            second_derivative = k/tau_estimate**2 - 2 * ts.sum() / tau_estimate**3 - 2*(N-k)*self.t_max / tau_estimate**3
            tau_estimate_error = np.sqrt(-1 / second_derivative)

            if print_stats:
                print(f'Estimated tau: {tau_estimate:.4f} +/- {tau_estimate_error:.4f}')

            return ts, tau_estimate, tau_estimate_error, term_1

    def print_stats(self, tau_estimates):
        print(f'Mean = {tau_estimates.mean():.4f} +/- {tau_estimates.std(ddof=1)/np.sqrt(len(tau_estimates)):.4f}')

    def estimate_stats(self, print_stats=True):
        tau_estimates = np.empty(self.n_runs)

        for i in range(self.n_runs):
            _, tau_estimates[i], _, _ = self.estimate()

        if print_stats==True:
            self.print_stats(tau_estimates)

        return tau_estimates

    def plot_estimate_by_cutoff(self):
        t_max_array = np.linspace(1, self.tau_true + 20, self.n_runs)

        tau_estimates_1 = self.estimate_stats()
    
        fig,ax = plt.subplots(figsize=(8,6))
        ax.plot(t_max_array, tau_estimates_1)
        plt.show()
    
    def plot_estimates(self):
        fig,ax = plt.subplots(figsize=(10,6))
        ts, tau_estimate_1, _, _ = self.estimate()
        hist_values, bin_edges, _ = plt.hist(ts, bins=int(np.sqrt(len(ts))))
        ax.vlines(self.tau_true, 0, 1.1 * hist_values.max(), colors='g', linestyles='--', label='True mean')
        ax.vlines(tau_estimate_1, 0, 1.1 * hist_values.max(), colors='b', linestyles='solid', label='Estimate (1)')
        ax.vlines(self.t_max, 0, 1.1 * hist_values.max(), colors='r', linestyles='solid', label=r'$t_{max}$')
        ax.legend()
        plt.show()

    def t_mean_below_k(self):
        return self.tau_true - self.t_max / (np.exp(self.t_max/self.tau_true) - 1)

    # @staticmethod
    # @np.vectorize(excluded={0, 1})
    # def py_hyper(a,b,z):
    #     return float(mpmath.hyper(a,b,z))

    # The integral of the continuous distribution below t_max
    @staticmethod
    def p(t_max, TAU):
        return 1 - np.exp(-t_max/TAU)

    def hypergeometric(self, p):
        #p = p[0]
        mpmath_func = mpmath.hyper([1, 1, 1 - self.n_samples], [2, 2], p / (p - 1))
        return float(mpmath_func)

    # The analytical result of the average tau estimate with known tau_true
    def average_tau_estimate(self):
        p = self.p(self.t_max, self.tau_true)
        h = self.hypergeometric(p)
        return self.t_mean_below_k() + (self.n_samples**2 * p * (1-p)**(self.n_samples-1) * h - 1) * self.t_max

    # Computes the parameter tau given an estimate for tau
    def tau_given_estimate(self, tau_estimate):
        func = lambda tau: tau - (1/(np.exp(self.t_max/tau) - 1)\
                                  - self.n_samples**2 * self.p(self.t_max, tau)\
                                                      * (1-self.p(self.t_max, tau))**(self.n_samples-1)\
                                                      * self.hypergeometric(self.p(self.t_max, tau)) + 1)\
                                  * self.t_max - tau_estimate

        try:
            res = optimize.newton(func, self.tau_true)
        except RuntimeError:
            return self.tau_true

        return res


def expectations_vs_analytic(n_samples_max, n_runs, tau_true, t_max):
    Ns = np.arange(1,n_samples_max)
    expectation_means = np.empty(len(Ns))
    analytic_mean = np.empty(len(Ns))
    
    for i in range(len(Ns)):
        if i % 10 == 0:
            print(f'{i} / {n_samples_max}')
        estimator_obj = Estimator(tau_true=tau_true, t_max=t_max, n_samples=Ns[i], n_runs=n_runs)
        expectations = np.empty(n_runs)

        for j in range(n_runs):
            _, _, _, expectations[j] = estimator_obj.estimate()

        expectation_means[i] = expectations.mean()
        analytic_mean[i] = estimator_obj.t_mean_below_k()

    plt.plot(Ns, expectation_means, label='Expectations')
    plt.plot(Ns, analytic_mean, label='Analytic')
    plt.xlabel(r'$N$')
    plt.ylabel('Mean values')
    plt.title(r'$t_{max}$' + f' = {t_max}, ' + r'$\tau_{true}$' + f' = {tau_true}')
    plt.legend(loc='best')
    plt.show()


def solve_empirical(tau_estimate, N, t_max):
    func = lambda tau: tau * N / (N - np.pi*np.exp(-t_max/tau)) - tau_estimate
    res = optimize.fsolve(func, tau_estimate)
    return res[0]


def plot_taus():
    n_plots = 4
    fig, ax = plt.subplots(n_plots,1, figsize=(8,18))
    n_samples_min = 5
    n_samples_max = 100
    Ns = np.arange(n_samples_min,n_samples_max, 1)
    t_max = 10

    for i in range(n_plots):
        tau_true = i+5
        taus_empirical = np.zeros(len(Ns))
        taus_estimates = np.zeros(len(Ns))

        errors_raw = np.zeros(len(Ns))
        errors_corrected = np.zeros(len(Ns))

        for ni, N in tqdm.tqdm(list(enumerate(Ns))):
            estimator_obj = Estimator(tau_true=tau_true, t_max=t_max, n_samples=N, n_runs=None)
            runs = 150
            for _ in range(runs):
                _, tau_estimate, _, _ = estimator_obj.estimate()
                if tau_estimate is None:
                    runs -= 1
                    continue

                # taus[ni] += estimator_obj.tau_given_estimate(tau_estimate)
                tau_corrected = solve_empirical(tau_estimate, N, t_max)
                taus_empirical[ni] += tau_corrected

                errors_raw[ni] += (tau_true - tau_estimate) ** 2
                errors_corrected[ni] += (tau_true - tau_corrected) ** 2

                taus_estimates[ni] += tau_estimate

            taus_empirical[ni] /= runs
            taus_estimates[ni] /= runs
            errors_raw[ni] = np.sqrt(errors_raw[ni] / runs)
            errors_corrected[ni] = np.sqrt(errors_corrected[ni] / runs)

        ax[i].plot(Ns, taus_estimates, label='Raw estimate')
        ax[i].plot(Ns, errors_raw, label='Raw Error')
        ax[i].plot(Ns, errors_corrected, label='Corrected Error')
        ax[i].plot(Ns, taus_empirical, ls='--', label='Empirical estimator')
        ax[i].plot(Ns, tau_true + Ns * 0, label=r'$\tau_{true}$')
        txt = f'Mean estimate = {taus_estimates.mean():.3f} '\
             + f'+/- {taus_estimates.std(ddof=1) / np.sqrt(len(taus_estimates)):.3f}'
        ax[i].text(n_samples_max/5, t_max+1, txt, c='r')
        ax[i].set(ylim=(0,t_max+3))
        ax[i].set_ylabel(r'$\tau$', size=14)
        ax[i].legend(loc='upper right')
    ax[n_plots-1].set_xlabel(r'$N$', size=14)
    plt.show()


if __name__ == '__main__':
    #expectations_vs_analytic(n_samples_max=10, n_runs=1000, tau_true=2, t_max=10)

    estimator_obj = Estimator(tau_true=1, t_max=4, n_samples=10000, n_runs=100)
    #tau_estimates, _ = estimator_obj.estimate_stats()
    _, tau_estimate, tau_estimate_std, _ = estimator_obj.estimate(print_stats=True)

    plot_taus()

