# PCWA
A highly parallel and fast event detector based on CWT transform.


## Requirements
- Python >= 3.8.5
- numpy >= 1.19.2
- scipy >= 1.6
- matplotlib >= 3.3.4
- h5py >= 2.10.0

## How to use PCWA
PCWA is designed as a Python class and requires initializing. Import pcwa and initiate a new instant:

```python
import pcwa as pcwa

pcwa_analyzer = pcwa.PCWA(parallel=False)
# pcwa_analyzer.show_wavelets = True
pcwa_analyzer.w, pcwa_analyzer.h = 1.5, 1.5
pcwa_analyzer.selectivity = 0.7
pcwa_analyzer.usescratchfile = False
```
properties can be set during or after initializing. A list of properties are as below:

## Properties
```python
dt = 1e5                               # sampling period of the signal in s
parallel = True                        # enable/disable multiprocessing 
Mcluster = True                        # enable/disable Maco-clustering
logscale = True                        # enable/disable logarithmic scale for scale-axis
wavelet = ['ricker']                   # list of wavelet function names
scales = [0.01e-3,0.1e-3,30]           # scale range and count in in s
selectivity = 0.5                      # minimum number of candidates in a valid micro-cluster
w = 2                                  # spreading factor in x-axis
h = 6                                  # spreading factor in y-axis (scale-axis)
trace = None                           # trace (data) variable. 1D numpy vector
events = []                            # list of detected events (valid after calling detect_events() function)
cwt = {}                               # dictionary of cwt coefficients
wavelets = {}                          # dictionary of generated scaled&normalized 1D wavelet arrays
show_wavelets = False                  # plot wavelet functions
update_cwt = True                      # if False, will use the current cwt coefficients to detect events to save time tuning threshold parameters
usescratchfile = False                 # stores cwt coefficients in the scarach file (hdf5 formatted) file
```

## Event Detection
After initializing, events can be detected by calling `detect_events()` method.

```python
events = pcwa_analyzer.detect_events(trace=binned_counts,wavelet=['msg-6','msg-7','msg-8'],scales=[0.1e-3,1.0e-3,50],threshold=3)
tpr, fdr = pcwa.tprfdr(label_dict["time"],events['time'],e=7e-3/1e-5,MS=0) # e is the tolerance of error for event location, here 7ms/0.01ms (in data points), 0.01ms is the bin size
```
some of pcwa parameters can overridden when calling `detect_events()` by passing the following parameters:
- `trace`:        overrides the trace
- `wavelet`:      overrides wavelet functions
- `scales`:       overrides scales

`threshold` is the only parameter required at each call.
