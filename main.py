import feather
import matplotlib.pyplot as plt
import numpy as np
import tqdm

PI = np.pi


def allan_deviation(data: np.ndarray, f: int, max_num_M: int = 100):
    """
    :param data: 1D data array
    :param f: data sample rate in Hz
    :param max_num_M: number of output points
    :return: (tau, allan_dev): tuple of result
             tau: array of tau values
             allan_dev: array of allan deviations
    """
    ts = 1.0 / f
    L = len(data)
    M_max = 2 ** np.floor(np.log2(L / 2))
    M = np.logspace(np.log10(1), np.log10(M_max), num=max_num_M)
    M = np.ceil(M)  # round up to integer
    M = np.unique(M)  # remove duplicate
    taus = M * ts  # compute `cluster durations` tau

    # compute allan variance
    allan_var = np.zeros(len(M))
    for i, mi in tqdm.tqdm(enumerate(M), desc='compute allan variance...'):
        two_Mi = int(2 * mi)
        mi = int(mi)
        allan_var[i] = np.sum(
            (data[two_Mi:L] - (2.0 * data[mi:L-mi]) + data[0:L-two_Mi]) ** 2
        )

    allan_var /= (2.0 * taus**2) * (L - (2.0 * M))
    return taus, np.sqrt(allan_var)


if __name__ == '__main__':
    file_path = './imu0.feather'
    freq = 400

    # Load csv into np array
    df = feather.read_dataframe(file_path)
    gyro_data_arr = df.iloc[:, 1:4].to_numpy()
    accel_data_arr = df.iloc[:, 4:].to_numpy()
    ts = 1.0 / freq

    # Separate into arrays
    gx = gyro_data_arr[:, 0] * (180.0 / PI)
    gy = gyro_data_arr[:, 1] * (180.0 / PI)
    gz = gyro_data_arr[:, 2] * (180.0 / PI)
    ax = accel_data_arr[:, 0] * (180.0 / PI)
    ay = accel_data_arr[:, 1] * (180.0 / PI)
    az = accel_data_arr[:, 2] * (180.0 / PI)

    # Calculate gyro angles
    gyro_theta_x = np.cumsum(gx) * ts
    gyro_theta_y = np.cumsum(gy) * ts
    gyro_theta_z = np.cumsum(gz) * ts

    # Calculate accel velocity
    accel_theta_x = np.cumsum(ax) * ts
    accel_theta_y = np.cumsum(ay) * ts
    accel_theta_z = np.cumsum(az) * ts

    # Compute allan deviations
    gyro_taux, gyro_adx = allan_deviation(gyro_theta_x, freq, max_num_M=100)
    gyro_tauy, gyro_ady = allan_deviation(gyro_theta_y, freq, max_num_M=100)
    gyro_tauz, gyro_adz = allan_deviation(gyro_theta_z, freq, max_num_M=100)
    accel_taux, accel_adx = allan_deviation(accel_theta_x, freq, max_num_M=100)
    accel_tauy, accel_ady = allan_deviation(accel_theta_y, freq, max_num_M=100)
    accel_tauz, accel_adz = allan_deviation(accel_theta_z, freq, max_num_M=100)

    #####################
    # Angle Random Walk #
    #####################
    # Find the index where the slope of the log-scaled Allan deviation is equal to the slope specified.
    slope = -0.5
    gyro_log_taux = np.log10(gyro_taux)
    gyro_log_tauy = np.log10(gyro_tauy)
    gyro_log_tauz = np.log10(gyro_tauz)

    gyro_log_adx = np.log10(gyro_adx)
    gyro_log_ady = np.log10(gyro_ady)
    gyro_log_adz = np.log10(gyro_adz)

    gyro_dlog_adx = np.diff(gyro_log_adx) / np.diff(gyro_log_taux)
    gyro_dlog_ady = np.diff(gyro_log_ady) / np.diff(gyro_log_tauy)
    gyro_dlog_adz = np.diff(gyro_log_adz) / np.diff(gyro_log_tauz)

    gyro_argmin_abs_x = np.argmin(np.abs(gyro_dlog_adx - slope))
    gyro_argmin_abs_y = np.argmin(np.abs(gyro_dlog_ady - slope))
    gyro_argmin_abs_z = np.argmin(np.abs(gyro_dlog_adz - slope))

    # Find the y-intercept of the line
    gyro_b_x = gyro_log_adx[gyro_argmin_abs_x] - slope * gyro_log_taux[gyro_argmin_abs_x]
    gyro_b_y = gyro_log_ady[gyro_argmin_abs_y] - slope * gyro_log_tauy[gyro_argmin_abs_y]
    gyro_b_z = gyro_log_adz[gyro_argmin_abs_z] - slope * gyro_log_tauz[gyro_argmin_abs_z]

    # Determine the angle random walk coefficient from the line
    gyro_log_Nx = slope * np.log(1) + gyro_b_x
    gyro_log_Ny = slope * np.log(1) + gyro_b_y
    gyro_log_Nz = slope * np.log(1) + gyro_b_z

    gyro_Nx = 10 ** gyro_log_Nx
    gyro_Ny = 10 ** gyro_log_Ny
    gyro_Nz = 10 ** gyro_log_Nz

    # Plot the result
    tauN = 1
    gyro_lineNx = gyro_Nx / np.sqrt(gyro_taux)
    gyro_lineNy = gyro_Ny / np.sqrt(gyro_tauy)
    gyro_lineNz = gyro_Nz / np.sqrt(gyro_tauz)

    #####################
    # Rate Random Walk #
    #####################
    # Find the index where the slope of the log-scaled Allan deviation is equal to the slope specified.
    slope = 0.5
    gyro_log_taux = np.log10(gyro_taux)
    gyro_log_tauy = np.log10(gyro_tauy)
    gyro_log_tauz = np.log10(gyro_tauz)

    gyro_log_adx = np.log10(gyro_adx)
    gyro_log_ady = np.log10(gyro_ady)
    gyro_log_adz = np.log10(gyro_adz)

    gyro_dlog_adx = np.diff(gyro_log_adx) / np.diff(gyro_log_taux)
    gyro_dlog_ady = np.diff(gyro_log_ady) / np.diff(gyro_log_tauy)
    gyro_dlog_adz = np.diff(gyro_log_adz) / np.diff(gyro_log_tauz)

    gyro_argmin_abs_x = np.argmin(np.abs(gyro_dlog_adx - slope))
    gyro_argmin_abs_y = np.argmin(np.abs(gyro_dlog_ady - slope))
    gyro_argmin_abs_z = np.argmin(np.abs(gyro_dlog_adz - slope))

    # Find the y-intercept of the line
    gyro_b_x = gyro_log_adx[gyro_argmin_abs_x] - slope * gyro_log_taux[gyro_argmin_abs_x]
    gyro_b_y = gyro_log_ady[gyro_argmin_abs_y] - slope * gyro_log_tauy[gyro_argmin_abs_y]
    gyro_b_z = gyro_log_adz[gyro_argmin_abs_z] - slope * gyro_log_tauz[gyro_argmin_abs_z]

    # Determine the rate random walk coefficient from the line
    gyro_log_Kx = slope * np.log10(3) + gyro_b_x
    gyro_log_Ky = slope * np.log10(3) + gyro_b_y
    gyro_log_Kz = slope * np.log10(3) + gyro_b_z

    gyro_Kx = 10 ** gyro_log_Kx
    gyro_Ky = 10 ** gyro_log_Ky
    gyro_Kz = 10 ** gyro_log_Kz

    # Plot the result
    tauK = 3
    gyro_lineKx = gyro_Kx * np.sqrt(gyro_taux / 3)
    gyro_lineKy = gyro_Ky * np.sqrt(gyro_tauy / 3)
    gyro_lineKz = gyro_Kz * np.sqrt(gyro_tauz / 3)

    ####################
    # Bias Instability #
    ####################
    # Find the index where the slope of the log-scaled Allan deviation is equal to the slope specified.
    slope = 0.0
    gyro_log_taux = np.log10(gyro_taux)
    gyro_log_tauy = np.log10(gyro_tauy)
    gyro_log_tauz = np.log10(gyro_tauz)

    gyro_log_adx = np.log10(gyro_adx)
    gyro_log_ady = np.log10(gyro_ady)
    gyro_log_adz = np.log10(gyro_adz)

    gyro_dlog_adx = np.diff(gyro_log_adx) / np.diff(gyro_log_taux)
    gyro_dlog_ady = np.diff(gyro_log_ady) / np.diff(gyro_log_tauy)
    gyro_dlog_adz = np.diff(gyro_log_adz) / np.diff(gyro_log_tauz)

    gyro_argmin_abs_x = np.argmin(np.abs(gyro_dlog_adx - slope))
    gyro_argmin_abs_y = np.argmin(np.abs(gyro_dlog_ady - slope))
    gyro_argmin_abs_z = np.argmin(np.abs(gyro_dlog_adz - slope))

    # Find the y-intercept of the line
    gyro_b_x = gyro_log_adx[gyro_argmin_abs_x] - slope * gyro_log_taux[gyro_argmin_abs_x]
    gyro_b_y = gyro_log_ady[gyro_argmin_abs_y] - slope * gyro_log_tauy[gyro_argmin_abs_y]
    gyro_b_z = gyro_log_adz[gyro_argmin_abs_z] - slope * gyro_log_tauz[gyro_argmin_abs_z]

    # Determine the bias instability coefficient from the line
    scfB = np.sqrt(2 * np.log(2) / PI)
    gyro_log_Bx = gyro_b_x - np.log10(scfB)
    gyro_log_By = gyro_b_y - np.log10(scfB)
    gyro_log_Bz = gyro_b_z - np.log10(scfB)

    gyro_Bx = 10 ** gyro_log_Bx
    gyro_By = 10 ** gyro_log_By
    gyro_Bz = 10 ** gyro_log_Bz

    # Plot the result
    tauBx = gyro_taux[gyro_argmin_abs_x]
    tauBy = gyro_tauy[gyro_argmin_abs_y]
    tauBz = gyro_tauz[gyro_argmin_abs_z]
    gyro_lineBx = gyro_Bx * scfB * np.ones(len(gyro_taux))
    gyro_lineBy = gyro_By * scfB * np.ones(len(gyro_tauy))
    gyro_lineBz = gyro_Bz * scfB * np.ones(len(gyro_tauz))

    # Plot all at once
    plt.figure()
    plt.title('Allan deviation')

    # plot angle random walk
    plt.plot(gyro_taux, gyro_adx, label='gx')
    plt.plot(gyro_tauy, gyro_ady, label='gy')
    plt.plot(gyro_tauz, gyro_adz, label='gz')
    plt.plot(gyro_taux, gyro_lineNx, ls='--', label=r'$\sigma_N$_gyroX')
    plt.plot(gyro_tauy, gyro_lineNy, ls='--', label=r'$\sigma_N$_gyroY')
    plt.plot(gyro_taux, gyro_lineNz, ls='--', label=r'$\sigma_N$_gyroZ')
    plt.plot(tauN, gyro_Nx, 'o')
    plt.plot(tauN, gyro_Ny, 'o')
    plt.plot(tauN, gyro_Nz, 'o')
    plt.text(tauN, gyro_Nx, 'N={:.4f}'.format(gyro_Nx))
    plt.text(tauN, gyro_Ny, 'N={:.4f}'.format(gyro_Ny))
    plt.text(tauN, gyro_Nz, 'N={:.4f}'.format(gyro_Nz))

    # plot rate random walk
    plt.plot(gyro_taux, gyro_adx, label='gx')
    plt.plot(gyro_tauy, gyro_ady, label='gy')
    plt.plot(gyro_tauz, gyro_adz, label='gz')
    plt.plot(gyro_taux, gyro_lineKx, ls='--', label=r'$\sigma_K$_gyroX')
    plt.plot(gyro_tauy, gyro_lineKy, ls='--', label=r'$\sigma_K$_gyroY')
    plt.plot(gyro_taux, gyro_lineKz, ls='--', label=r'$\sigma_K$_gyroZ')
    plt.plot(tauK, gyro_Kx, 'o')
    plt.plot(tauK, gyro_Ky, 'o')
    plt.plot(tauK, gyro_Kz, 'o')
    plt.text(tauK, gyro_Kx, 'K={:.4f}'.format(gyro_Kx))
    plt.text(tauK, gyro_Ky, 'K={:.4f}'.format(gyro_Ky))
    plt.text(tauK, gyro_Kz, 'K={:.4f}'.format(gyro_Kz))

    # plot bias instability
    plt.plot(gyro_taux, gyro_adx, label='gx')
    plt.plot(gyro_tauy, gyro_ady, label='gy')
    plt.plot(gyro_tauz, gyro_adz, label='gz')
    plt.plot(gyro_taux, gyro_lineBx, ls='--', label=r'$\sigma_B$_gyroX')
    plt.plot(gyro_tauy, gyro_lineBy, ls='--', label=r'$\sigma_B$_gyroY')
    plt.plot(gyro_taux, gyro_lineBz, ls='--', label=r'$\sigma_B$_gyroZ')
    plt.plot(tauBx, scfB * gyro_Bx, 'o')
    plt.plot(tauBy, scfB * gyro_By, 'o')
    plt.plot(tauBz, scfB * gyro_Bz, 'o')
    plt.text(tauBx, scfB * gyro_Bx, 'B={:.4f}'.format(gyro_Bx))
    plt.text(tauBy, scfB * gyro_By, 'B={:.4f}'.format(gyro_By))
    plt.text(tauBz, scfB * gyro_Bz, 'B={:.4f}'.format(gyro_Bz))

    plt.xlabel(r'$\tau$ [sec]')
    plt.ylabel(r'$\sigma$($\tau$) [deg/sec]')
    plt.grid(True, which='both', ls='-', color='0.65')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.show()