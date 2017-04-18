Collection of functions to help in the use of Keras.  

Recommended usage:  
```python
import messlkeras as mk
```  
Currently does not support `from messlkeras import *` .  

The following functions will be available in the module's namespace, ex: `mk.prep_data_for_keras(...)`

```python
prep_list_for_keras(data_dir, reg_exp, verbose=False)
    ### prepare the list of .mat files containing the data, which will then be fed to prep_data_for_keras
    # data_dir is the file directory containing the data (which could be in subdirectories)
    # reg_exp is the regular expression that identifies the directories containing the data to use:
    # eg: for noisy spectrogram dev set use "messl-spects-noisy.*dt" 
    # so only .mat in directories with that pattern will beadded to the list

	# returns a list
    

prep_data_for_keras(file_list, input_shape=(-1, 50, 513), start=0, chan2keep='nan', time_limit=180, verbose=False)
    ### prepares the data for Keras, using CHIME3 data only!
    # file_list details with .mat files to load (created by prep_list_for_keras)
    # input_shape will define the shape of the data: (sample_num, input_length, features) (must all be positive)
    # input_shape[0]=-1 means that b default it will load all the files in the file_list, time permitting
    # start=n allows the user to start later in the lists
    # time_limit puts on cap on how long the process should take, in seconds. default is 3 minutes
    # chan2keep: in cases where the data is 2 channels, needs to specify which to keep. Should have value 0 or 1. Not required in other cases.

    # returns a numpy.ndarray of shape input_shape

```
