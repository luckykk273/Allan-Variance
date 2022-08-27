from allan_variance_utils import *
import feather


if __name__ == '__main__':
    file_path = './imu0.feather'

    # Load csv into np array
    df = feather.read_dataframe(file_path)  # also, you can read in csv file with pandas
    freq = 400  # define your data's sample rate

    # Transform gyro unit from [rad/s] to [deg/s]
    df.iloc[:, 1:4] = df.iloc[:, 1:4] * 180 / PI
    gyro_x = df.iloc[:, 1].to_numpy()

    # Pass data into function
    # Note that the function only accept one axis;
    # tau_x, allan_var_x = allan_variance(data=gyro_x, f=freq, max_clusters=200)
    # allan_dev_x = np.sqrt(allan_var_x)
    # plot_result(tau_x, allan_dev_x)

    # Another version to compute Allan variance
    allan_vars_x = allan_variance2(data=gyro_x, f=freq)
    periods_x, allan_var_x = allan_vars_x[:, 0], allan_vars_x[:, 1]
    allan_dev_x = np.sqrt(allan_var_x)
    plot_result2(periods_x, allan_dev_x)
