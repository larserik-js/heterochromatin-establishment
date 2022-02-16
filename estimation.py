import numpy as np
import matplotlib.pyplot as plt
import mpmath
import tqdm as tqdm
from scipy import optimize
r = np.random

class Estimator:
    def __init__(self, TAU_TRUE, T_MAX, N_SAMPLES, N_RUNS):
        self.TAU_TRUE = TAU_TRUE
        self.T_MAX = T_MAX
        # Number of samples
        self.N_SAMPLES = N_SAMPLES
        # Number of runs
        self.N_RUNS = N_RUNS

    def estimate(self, print_stats=False):
        # All sampled data
        data = r.exponential(scale=self.TAU_TRUE, size=self.N_SAMPLES)
        Is = (data < self.T_MAX)

        # No data below T_MAX
        if Is.sum() == 0:
            return None, None, None, None
        else:
            # Data sampled from continuous distribution
            ts = data[Is]
            k = len(ts)
            N = len(data)

            # Terms for the estimate of tau
            TERM_1 = ts.mean()
            TERM_2 = (N / k - 1) * self.T_MAX

            tau_estimate = TERM_1 + TERM_2

            # The error
            second_derivative = k/tau_estimate**2 - 2 * ts.sum() / tau_estimate**3 - 2*(N-k)*self.T_MAX / tau_estimate**2
            tau_estimate_error = np.sqrt(-1 / second_derivative)

            if print_stats == True:
                print(f'Estimated tau: {tau_estimate:.4f} +/- {tau_estimate_error:.4f}')

            return ts, tau_estimate, tau_estimate_error, TERM_1

    def print_stats(self, tau_estimates):
        print(f'Mean = {tau_estimates.mean():.4f} +/- {tau_estimates.std(ddof=1)/np.sqrt(len(tau_estimates)):.4f}')

    def estimate_stats(self, print_stats=True):
        tau_estimates = np.empty(self.N_RUNS)

        for i in range(self.N_RUNS):
            _, tau_estimates[i], _, _ = self.estimate()

        if print_stats==True:
            self.print_stats(tau_estimates)

        return tau_estimates


    def plot_estimate_by_cutoff(self):
        t_max_array = np.linspace(1, self.TAU_TRUE + 20, self.N_RUNS)

        tau_estimates_1 = self.estimate_stats()
    
        fig,ax = plt.subplots(figsize=(8,6))
        ax.plot(t_max_array, tau_estimates_1)
        plt.show()
    
    def plot_estimates(self):
        fig,ax = plt.subplots(figsize=(10,6))
        ts, tau_estimate_1, _, _ = self.estimate()
        hist_values, bin_edges, _ = plt.hist(ts, bins=int(np.sqrt(len(ts))))
        ax.vlines(self.TAU_TRUE, 0, 1.1 * hist_values.max(), colors='g', linestyles='--', label='True mean')
        ax.vlines(tau_estimate_1, 0, 1.1 * hist_values.max(), colors='b', linestyles='solid', label='Estimate (1)')
        ax.vlines(self.T_MAX, 0, 1.1 * hist_values.max(), colors='r', linestyles='solid', label=r'$t_{max}$')
        ax.legend()
        plt.show()

    def t_mean_below_k(self):
        return self.TAU_TRUE - self.T_MAX / (np.exp(self.T_MAX/self.TAU_TRUE) - 1)

    # @staticmethod
    # @np.vectorize(excluded={0, 1})
    # def py_hyper(a,b,z):
    #     return float(mpmath.hyper(a,b,z))

    # The integral of the continuous distribution below T_MAX
    @staticmethod
    def p(T_MAX, TAU):
        return 1 - np.exp(-T_MAX/TAU)

    def hypergeometric(self, p):
        #p = p[0]
        mpmath_func = mpmath.hyper([1, 1, 1 - self.N_SAMPLES], [2, 2], p / (p - 1))
        return float(mpmath_func)

    # The analytical result of the average tau estimate with known tau_true
    def average_tau_estimate(self):
        p = self.p(self.T_MAX, self.TAU_TRUE)
        h = self.hypergeometric(p)
        return self.t_mean_below_k() + (self.N_SAMPLES**2 * p * (1-p)**(self.N_SAMPLES-1) * h - 1) * self.T_MAX

    # Computes the parameter tau given an estimate for tau
    def tau_given_estimate(self, tau_estimate):
        func = lambda tau: tau - (
                                  1/(np.exp(self.T_MAX/tau) - 1)\
                                  - self.N_SAMPLES**2 * self.p(self.T_MAX, tau)\
                                                      * (1-self.p(self.T_MAX, tau))**(self.N_SAMPLES-1)\
                                                      * self.hypergeometric(self.p(self.T_MAX, tau)) + 1)\
                                  * self.T_MAX - tau_estimate

        try:
            res = optimize.newton(func, self.TAU_TRUE)
        except RuntimeError:
            return self.TAU_TRUE

        return res

def expectations_vs_analytic(N_SAMPLES_MAX, N_RUNS, TAU_TRUE, T_MAX):
    Ns = np.arange(1,N_SAMPLES_MAX)
    expectation_means = np.empty(len(Ns))
    analytic_mean = np.empty(len(Ns))
    
    for i in range(len(Ns)):
        if i % 10 == 0:
            print(f'{i} / {N_SAMPLES_MAX}')
        estimator_obj = Estimator(TAU_TRUE=TAU_TRUE, T_MAX=T_MAX, N_SAMPLES=Ns[i], N_RUNS=N_RUNS)
        expectations = np.empty(N_RUNS)

        for j in range(N_RUNS):
            _, _, _, expectations[j] = estimator_obj.estimate()

        expectation_means[i] = expectations.mean()
        analytic_mean[i] = estimator_obj.t_mean_below_k()

    plt.plot(Ns, expectation_means, label='Expectations')
    plt.plot(Ns, analytic_mean, label='Analytic')
    plt.xlabel(r'$N$')
    plt.ylabel('Mean values')
    plt.title(r'$t_{max}$' + f' = {T_MAX}, ' + r'$\tau_{true}$' + f' = {TAU_TRUE}')
    plt.legend(loc='best')
    plt.show()

def solve_empirical(tau_estimate, N, T_MAX):
    func = lambda tau: tau * N / (N - np.pi*np.exp(-T_MAX/tau)) - tau_estimate
    res = optimize.fsolve(func, tau_estimate)
    return res[0]

def plot_taus():
    n_plots = 4
    fig, ax = plt.subplots(n_plots,1, figsize=(8,18))
    N_SAMPLES_MIN = 5
    N_SAMPLES_MAX = 100
    Ns = np.arange(N_SAMPLES_MIN,N_SAMPLES_MAX, 1)
    T_MAX = 10

    for i in range(n_plots):
        TAU_TRUE = i+5
        taus = np.zeros(len(Ns))
        taus_empirical = np.zeros(len(Ns))
        taus_estimates = np.zeros(len(Ns))

        errors_raw = np.zeros(len(Ns))
        errors_corrected = np.zeros(len(Ns))

        for ni, N in tqdm.tqdm(list(enumerate(Ns))):
            estimator_obj = Estimator(TAU_TRUE=TAU_TRUE, T_MAX=T_MAX, N_SAMPLES=N, N_RUNS=None)
            runs = 150
            for _ in range(runs):
                _, tau_estimate, _, _ = estimator_obj.estimate()
                if tau_estimate is None:
                    runs -= 1
                    continue

                # taus[ni] += estimator_obj.tau_given_estimate(tau_estimate)
                tau_corrected = solve_empirical(tau_estimate, N, T_MAX)
                taus_empirical[ni] += tau_corrected

                errors_raw[ni] += (TAU_TRUE - tau_estimate) ** 2
                errors_corrected[ni] += (TAU_TRUE - tau_corrected) ** 2

                taus_estimates[ni] += tau_estimate

            taus[ni] /= runs
            taus_empirical[ni] /= runs
            taus_estimates[ni] /= runs
            errors_raw[ni] = np.sqrt(errors_raw[ni] / runs)
            errors_corrected[ni] = np.sqrt(errors_corrected[ni] / runs)

        #print(taus[:10])
        #print(taus_empirical[:10])

        ax[i].plot(Ns, taus_estimates, label='Raw estimate')
        ax[i].plot(Ns, taus, label='Estimator')
        ax[i].plot(Ns, errors_raw, label='Raw Error')
        ax[i].plot(Ns, errors_corrected, label='Corrected Error')
        ax[i].plot(Ns, taus_empirical, ls='--', label='Empirical estimator')
        ax[i].plot(Ns, TAU_TRUE + Ns * 0, label=r'$\tau_{true}$')
        txt = f'Mean estimate = {taus.mean():.3f} +/- {taus.std(ddof=1) / np.sqrt(len(taus)):.3f}'
        ax[i].text(N_SAMPLES_MAX/5, T_MAX+1, txt, c='r')
        ax[i].set(ylim=(0,T_MAX+3))
        ax[i].set_ylabel(r'$\tau$', size=14)
        ax[i].legend(loc='upper right')
    ax[n_plots-1].set_xlabel(r'$N$', size=14)
    plt.show()

if __name__ == '__main__':
    #expectations_vs_analytic(N_SAMPLES_MAX=10, N_RUNS=1000, TAU_TRUE=2, T_MAX=10)

    estimator_obj = Estimator(TAU_TRUE=1, T_MAX=4, N_SAMPLES=1000, N_RUNS=10000)
    #tau_estimates, _ = estimator_obj.estimate_stats()
    _, tau_estimate, tau_estimate_std, _ = estimator_obj.estimate(print_stats=True)

    #plot_taus()

