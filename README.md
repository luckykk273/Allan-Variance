# Allan Variance
Implement two versions for calculating Allan Variance, with some mathematical comments. The first version is implemented based on MATLAB documentation; the second version is a reimplementation of [ori-drs/allan_variance_ros](https://github.com/ori-drs/allan_variance_ros) in Python. Both two versions have been tested, and their output match the respective reference implementations.

## Preface
When doing noise analysis for inertial measurement unit(IMU), Allan variance is one of the most useful methods. There are some libraries wrote in MATLAB, Python, or C++. Most of these libraries doesn't build bridge between theory and implementation of code. This project's aims to give some intuitive explanation for some part of code when computing Allan variance.  

## Data
Inertial sensor noise analysis are suggested to keep sensor static for at least **3hrs**. The example data are too large to upload to GitHub(though the file has been compressed). And there are some technical problems to upload data with Git Large File Storage. Please go to the url provided to download the [IMU data](https://drive.google.com/file/d/1W6b9GrQ47dlNgfJLtvdxJ0_yoMjOvydz/view?usp=sharing).

## Requirement
All required packages are listed in `requirements.txt`. Please install the necessary libraries with the command:  
```bash
pip install -r requirements.txt
```

## Usage
Feel free to load the IMU data collected yourself and pass into the function(both two versions accept only one axis of data). In `main.py`, there is a simple example to follow: 
```Python
import feather
from allan_variance_utils import *


file_path = './imu0.feather'

# Load csv into np array
df = feather.read_dataframe(file_path)  # also, you can read in csv file with pandas
freq = 400  # define your data's sample rate

# Transform gyro unit from [rad/s] to [deg/s]
df.iloc[:, 1:4] = df.iloc[:, 1:4] * 180 / PI
gyro_x = df.iloc[:, 1].to_numpy()

# Pass data into function
# Note that the function only accept one axis;
tau_x, allan_var_x = allan_variance(data=gyro_x, f=freq, max_clusters=200)
allan_dev_x = np.sqrt(allan_var_x)
plot_result(tau_x, allan_dev_x)

# Another version to compute Allan variance
allan_vars_x = allan_variance2(data=gyro_x, f=freq)
periods_x, allan_var_x = allan_vars_x[:, 0], allan_vars_x[:, 1]
allan_dev_x = np.sqrt(allan_var_x)
plot_result2(periods_x, allan_dev_x)
```

![version1](https://github.com/luckykk273/Allan-Variance/blob/main/example_gyro_x.png)
![version2](https://github.com/luckykk273/Allan-Variance/blob/main/example_gyro_x2.png)

NOTE: When you compare the results got from these two versions, you will find slightly different between them because **different data range are used** when fitting the line. For instance, when finding out white noise:  

In version 1, slopes are computed at all tau and it finds the index most closed to -0.5:
```Python
log_tau = np.log10(tau)
log_allan_dev = np.log10(allan_dev)
dlog_allan_dev = np.diff(log_allan_dev) / np.diff(log_tau)

slope_arw = -0.5
argmin_abs_i = np.argmin(np.abs(dlog_allan_dev - slope_arw))  # index that are most closed to -0.5

# Find the y-intercept of the line
intercept = log_allan_dev[argmin_abs_i] - slope_arw * log_tau[argmin_abs_i]
```
In version 2, it fits data from the beginning to where periods equal to 10:
```Python
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

# White noise(velocity/angle random walk)
bp_wn = np.where(periods == 10)[0][0]  # white noise break point for short.
wn, fit_func_wn = _fit_intercept(periods[0:bp_wn], allan_dev[0:bp_wn], -0.5, 1.0)
```

## Reference
### Theory
1. [Wiki - Allan Variance](https://en.wikipedia.org/wiki/Allan_variance)  
2. [N. El-Sheimy, H. Hou and X. Niu, "Analysis and Modeling of Inertial Sensors Using Allan Variance," in IEEE Transactions on Instrumentation and Measurement, vol. 57, no. 1, pp. 140-149, Jan. 2008, doi: 10.1109/TIM.2007.908635.](https://ieeexplore.ieee.org/document/4404126)  
3. ["IEEE Standard Specification Format Guide and Test Procedure for Single-Axis Laser Gyros," in IEEE Std 647-2006 (Revision of IEEE Std 647-1995) , vol., no., pp.1-96, 18 Sept. 2006, doi: 10.1109/IEEESTD.2006.246241.](https://ieeexplore.ieee.org/document/1706054)  
4. [Pupo, Leslie Barreda. “Characterization of errors and noises in MEMS inertial sensors using Allan variance method.” (2016).](https://www.semanticscholar.org/paper/Characterization-of-errors-and-noises-in-MEMS-using-Pupo/b8d4eaa1ed06274534ebe843b2e5c881ac380dd9)  
5. [Introduction to Allan Variance—Non-overlapping and Overlapping Allan Variance](https://www.allaboutcircuits.com/technical-articles/intro-to-allan-variance-analysis-non-overlapping-and-overlapping-allan-variance/)

If someone is a newbie to IMU or Allan variance(or maybe you are confused about terms mentioned in this repository), please refer to the [Introduction to Simulating IMU Measurements](https://www.mathworks.com/help/nav/ug/introduction-to-simulating-imu-measurements.html).

### Implementation
- [Inertial Sensor Noise Analysis Using Allan Variance](https://www.mathworks.com/help/nav/ug/inertial-sensor-noise-analysis-using-allan-variance.html) in MATLAB documentation.  
- [ori-drs/allan_variance_ros](https://github.com/ori-drs/allan_variance_ros)  

## Contact
Welcome to contact me for any further question through my email:  
luckykk273@gmail.com
