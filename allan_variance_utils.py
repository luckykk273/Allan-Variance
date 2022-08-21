import matplotlib.pyplot as plt
import numpy as np


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
    print('Angle random walk:', n)

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

    # Plot angle random walk
    plt.plot(tau, line_n, ls='--', label=r'$\sigma_N$')
    plt.plot(tau_n, n, 'o')
    plt.text(tau_n, n, 'N={:.4f}'.format(n))

    # Plot rate random walk
    plt.plot(tau, line_k, ls='--', label=r'$\sigma_K$')
    plt.plot(tau_k, k, 'o')
    plt.text(tau_k, k, 'K={:.4f}'.format(k))

    # plot bias instability
    plt.plot(tau, line_b, ls='--', label=r'$\sigma_B$')
    plt.plot(tau_b, scf_b * b, 'o')
    plt.text(tau_b, scf_b * b, 'B={:.4f}'.format(b))

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

    :param data: 1D data array (unit: gyro [deg/s], accel [m/s^2])
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
            (data_integral[2*mi: n] - 2 * data_integral[mi: n - mi] + data_integral[0: n - 2*mi])**2
        )
    allan_var /= (2 * tau**2) * (n - 2*m)
    return tau, allan_var
