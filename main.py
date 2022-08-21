from allan_variance_utils import *
import feather


if __name__ == '__main__':
    file_path = './imu0.feather'

    # Load csv into np array
    df = feather.read_dataframe(file_path)
    freq = 400

    # Transform gyro unit from [rad/s] to [deg/s]
    df.iloc[:, 1:4] = df.iloc[:, 1:4] * 180 / PI
    gx = df.iloc[:, 1].to_numpy()

    tau_x, allan_var_x = allan_variance(data=gx, f=freq)
    allan_dev_x = np.sqrt(allan_var_x)
    plot_result(tau_x, allan_dev_x)
