# Continuous-time representation for signal decoding

In the signal decoding tasks, we work with multidimensional time series, which are a discretization of a continuous process. The latest works in neural ODE illustrate a possibility to work with recurrent neural networks as with differential equations.

This work addresses such applications as change of sampling rate and handling missed or irregular data. It becomes possible if we represent our signal as a continuous in time function. This approach is relevant for signals from various wearable devices: accelerometers, heart rate monitors, devices for picking up brain signals such as electroencephalograms or electrocorticograms.

The main result of this work is an algorithm which allows working with a signal as if it was a continuous function. We also consider different applications of this algorithm and propose to do further research on expanding the continuity of time to the continuity of space.
