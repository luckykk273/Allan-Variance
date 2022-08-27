import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from scipy.optimize import curve_fit

PI = np.pi


def plot_result(tau: np.ndarray, allan_dev: np.ndarray):
    log_tau = np.log10(tau)
    log_allan_dev = np.log10(allan_dev)
    dlog_allan_dev = np.diff(log_allan_dev) / np.diff(log_tau)

    #####################
    # Angle random walk #
    #####################
    slope_arw = -0.5
    argmin_abs_i = np.argmin(np.abs(dlog_allan_dev - slope_arw))

    # Find the y-intercept of the line
    intercept = log_allan_dev[argmin_abs_i] - slope_arw * log_tau[argmin_abs_i]

    # Determine the angle random walk coefficient from the line
    log_n = slope_arw * np.log(1) + intercept
    n = 10 ** log_n
    print('White noise(random walk):', n)

    # Plot the result
    tau_n = 1
    line_n = n / np.sqrt(tau)

    #####################
    # Rate Random Walk #
    #####################
    # Find the index where the slope of the log-scaled Allan deviation is equal to the slope specified.
    slope_rrw = 0.5
    argmin_abs_i = np.argmin(np.abs(dlog_allan_dev - slope_rrw))

    # Find the y-intercept of the line
    intercept = log_allan_dev[argmin_abs_i] - slope_rrw * log_tau[argmin_abs_i]

    # Determine the rate random walk coefficient from the line
    log_k = slope_rrw * np.log10(3) + intercept
    k = 10 ** log_k
    print('Rate Random Walk:', k)

    # Plot the result
    tau_k = 3
    line_k = k * np.sqrt(tau / 3)

    ####################
    # Bias Instability #
    ####################
    # Find the index where the slope of the log-scaled Allan deviation is equal to the slope specified.
    slope_bi = 0.0
    argmin_abs_i = np.argmin(np.abs(dlog_allan_dev - slope_bi))

    # Find the y-intercept of the line
    intercept = log_allan_dev[argmin_abs_i] - slope_bi * log_tau[argmin_abs_i]

    # Determine the bias instability coefficient from the line
    scf_b = np.sqrt(2 * np.log(2) / PI)
    log_b = intercept - np.log10(scf_b)
    b = 10 ** log_b
    print('Bias Instability:', b)

    # Plot the result
    tau_b = tau[argmin_abs_i]
    line_b = b * scf_b * np.ones(len(tau))

    ########
    # Plot #
    ########
    plt.figure()
    plt.title('Allan deviation')

    # Plot allan deviation
    plt.plot(tau, allan_dev)

    # Plot white noise(velocity/angle random walk)
    plt.plot(tau, line_n, ls='--', label=r'$\sigma_N$')
    plt.plot(tau_n, n, 'o')
    plt.text(tau_n, n, 'N={:.6f}'.format(n))

    # Plot rate random walk
    plt.plot(tau, line_k, ls='--', label=r'$\sigma_K$')
    plt.plot(tau_k, k, 'o')
    plt.text(tau_k, k, 'K={:.6f}'.format(k))

    # plot bias instability
    plt.plot(tau, line_b, ls='--', label=r'$\sigma_B$')
    plt.plot(tau_b, scf_b * b, 'o')
    plt.text(tau_b, scf_b * b, 'B={:.6f}'.format(b))

    plt.xlabel(r'$\tau_s$')
    plt.ylabel(r'$\sigma(\tau)$')
    plt.grid(True, which='both', ls='-', color='0.65')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.show()


def allan_variance(data: np.ndarray, f: int, max_clusters: int = 100):
    """
    The function to compute Allan variance.

    :param data: 1D numpy array (unit: gyro [deg/s], accel [m/s^2])
    :param f:  data sample rate [Hz]
    :param max_clusters: maximum number of clusters
    :return: (tau, allan_var): tuple of result
             tau: array of tau values
             allan_var: array of allan variance
    """
    tau0 = 1 / f  # sample period [s]
    data_integral = np.cumsum(data) * tau0
    n = len(data_integral)  # number of sample data
    m_max = 2 ** np.floor(np.log2(n / 2))
    m = np.logspace(np.log10(1), np.log10(m_max), num=max_clusters)
    m = np.ceil(m)
    m = np.unique(m)
    m = m.astype(int)
    tau = m * tau0  # cluster duration

    # Compute allan variance
    allan_var = np.zeros(len(m))
    for i, mi in enumerate(m):
        allan_var[i] = np.sum(
            (data_integral[2 * mi: n] - 2 * data_integral[mi: n - mi] + data_integral[0: n - 2 * mi]) ** 2
        )
    allan_var /= (2 * tau ** 2) * (n - 2 * m)
    return tau, allan_var


def plot_result2(periods: np.ndarray, allan_dev: np.ndarray):
    def _linear_func(x, a, b):
        return a * x + b

    def _fit_intercept(x, y, a, tau):
        log_x, log_y = np.log(x), np.log(y)
        # a in range [m, m+0.001]; b in range [-Inf, Inf]
        coefs, _ = curve_fit(_linear_func, log_x, log_y, bounds=([a, -np.inf], [a + 0.001, np.inf]))
        poly = np.poly1d(coefs)
        print('Fitting polynomial equation:', np.poly1d(poly))
        y_fit = lambda x: np.exp(poly(np.log(x)))
        return y_fit(tau), y_fit

    def _predict(tau, q, wn, bi, rr, ramp):
        A = np.array([3 / tau**2, 1 / tau, 2 * np.log(2) / np.pi, tau / 3, tau**2 / 2])
        params = np.array([q, wn, bi, rr, ramp]) ** 2
        return np.sqrt(A.dot(params))

    # White noise(velocity/angle random walk)
    bp_wn = np.where(periods == 10)[0][0]  # white noise break point for short.
    wn, fit_func_wn = _fit_intercept(periods[0:bp_wn], allan_dev[0:bp_wn], -0.5, 1.0)
    print('White noise(random walk):', wn)

    # Rate random walk
    rr, fit_func_rr = _fit_intercept(periods, allan_dev, 0.5, 3.0)
    print('Rate random walk:', rr)

    # Bias instability
    min_dev = np.min(allan_dev)
    argmin_dev = np.argmin(allan_dev)
    print('Bias instability:', min_dev)

    # Plot result
    fig = plt.figure()

    # Plot allan deviation
    plt.plot(periods, allan_dev)

    # Plot white noise(random walk)
    plt.plot(periods, fit_func_wn(periods), ls='--', label=r'$\sigma_N$')
    plt.plot(1.0, wn, 'o')
    plt.text(1.0, wn, 'N={:.6f}'.format(wn))

    # Plot rate random walk
    plt.plot(periods, fit_func_rr(periods), ls='--', label=r'$\sigma_K$')
    plt.plot(3.0, rr, 'o')
    plt.text(3.0, rr, 'N={:.6f}'.format(rr))

    # Plot bias instability
    plt.plot(periods, [min_dev]*len(periods), ls='--', label=r'$\sigma_B$')
    plt.plot(periods[argmin_dev], min_dev, 'o')
    plt.text(periods[argmin_dev], min_dev, 'B={:.6f}'.format(min_dev))

    # Plot fitted model
    fitted_model = _predict(periods, 0, wn, min_dev, rr, 0)
    plt.plot(periods, fitted_model, label='fitted model')

    plt.xlabel(r'$\tau_s$')
    plt.ylabel(r'$\sigma(\tau)$')
    plt.grid(True, which='both', ls='-', color='0.65')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.show()


@njit
def allan_variance2(data: np.ndarray, f: int, verbose: int = 0):
    """
    A much more intuitive understood function to compute Allan variance.
    NOTE: This function runs much slower(because this function doesn't utilize vectorization),
          so we use numba to accelerate.

    :param data: 1D numpy array (unit: gyro [deg/s], accel [m/s^2])
    :param f: data sample rate [Hz]
    :param verbose: show process of info. if verbose > 0
    :return: allan_variances: 2D numpy array which shape is (m, 2)
             Allan variances from t0, 2t0, ..., to mt0 with corresponding periods
    """
    allan_variances = []
    for period in range(1, 10000):
        avgs = []
        period_time = 0.1 * period
        bin_size = 0
        avg = 0
        max_bin_size = period_time * f

        # Compute averages from t0, 2t0, ..., to mt0
        for e in data:
            avg += e
            bin_size += 1
            if bin_size >= max_bin_size:
                avg /= bin_size
                avgs.append(avg)
                avg = 0
                bin_size = 0

        num_avgs = len(avgs)

        if verbose > 0:
            print('Compute', num_avgs, 'averages for period', period_time)

        # Compute Allan variance
        allan_var = 0
        for k in range(num_avgs - 1):
            allan_var += (avgs[k + 1] - avgs[k]) ** 2

        allan_var /= 2 * (num_avgs - 1)
        allan_variances.append((period_time, allan_var))

    return np.array(allan_variances)

