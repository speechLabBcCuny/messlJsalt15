Collection of functions to help in the use of Keras.  

Recommended usage:  
```python
import messlkeras as mk
```  
Currently does not support `from messlkeras import *` .  

The following functions will be available in the module's namespace, ex: `mk.prep_data_SM(...)`

```python
def prep_data_SM(spects_list, masks_list, input_shape=(100, 50, 513), start=0):
    ### prepares the data for Keras
    # masks_list, spects_list should have the corresponding filenames in the same order
    # input_shape will define the shape of the data: (sample_num, input_length, features) sample_num can be -1
    # start=n allows the user to start later in the lists    
    # the spectrogram data will be normalized to [-1..1]
```
