# Allan Variance

## Preface
When doing noise analysis for inertial measurement unit(IMU), 
Allan variance is one of the most useful methods.  
There are some libraries wrote in MATLAB, Python, or C++.  
Most of these libraries doesn't build bridge between theory and implementation of code.  
This project's aims to give some intuitive explanation for some part of code when computing Allan variance.


## Data
Inertial sensor noise analysis are suggested to keep sensor static for at least **3hrs**.
The example data are too large to upload to GitHub(though the file has been compressed).
And there are some technical problems to upload data with Git Large File Storage.
Please go to the url below to download the data:  
[IMU data](https://drive.google.com/file/d/1W6b9GrQ47dlNgfJLtvdxJ0_yoMjOvydz/view?usp=sharing)

## Requirement
All requirement packages are listed in requirements.txt.  
Please go to the directory where the repository clone and type the command below to install the necessary libraries:  
```pip install -r requirements.txt```

## Usage
Feel free to load the IMU data collected yourself and pass into the function.  
(Note that the function only accept one axis)  
In main.py, there is an easy example to follow: 
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
tau_x, allan_var_x = allan_variance(data=gyro_x, f=freq)
allan_dev_x = np.sqrt(allan_var_x)
plot_result(tau_x, allan_dev_x)
```

![image](https://github.com/luckykk273/Allan-Variance/blob/main/example_gyro_x.png)

## Reference
Most of the code refer to the [Inertial Sensor Noise Analysis Using Allan Variance](https://www.mathworks.com/help/nav/ug/inertial-sensor-noise-analysis-using-allan-variance.html) in MATLAB documentation.  
You can just see this project as a Python version.  
If someone has interest in mathematical theory of Allan variance, some resources are provided here:  
1. [Wiki - Allan Variance](https://en.wikipedia.org/wiki/Allan_variance)  
2. [N. El-Sheimy, H. Hou and X. Niu, "Analysis and Modeling of Inertial Sensors Using Allan Variance," in IEEE Transactions on Instrumentation and Measurement, vol. 57, no. 1, pp. 140-149, Jan. 2008, doi: 10.1109/TIM.2007.908635.](https://ieeexplore.ieee.org/document/4404126)  
3. ["IEEE Standard Specification Format Guide and Test Procedure for Single-Axis Laser Gyros," in IEEE Std 647-2006 (Revision of IEEE Std 647-1995) , vol., no., pp.1-96, 18 Sept. 2006, doi: 10.1109/IEEESTD.2006.246241.](https://ieeexplore.ieee.org/document/1706054)  
4. [Pupo, Leslie Barreda. “Characterization of errors and noises in MEMS inertial sensors using Allan variance method.” (2016).](https://www.semanticscholar.org/paper/Characterization-of-errors-and-noises-in-MEMS-using-Pupo/b8d4eaa1ed06274534ebe843b2e5c881ac380dd9)  
5. [Introduction to Allan Variance—Non-overlapping and Overlapping Allan Variance](https://www.allaboutcircuits.com/technical-articles/intro-to-allan-variance-analysis-non-overlapping-and-overlapping-allan-variance/)

## Contact
Welcome to contact me for any further question, below is my gmail address:  
luckykk273@gmail.com