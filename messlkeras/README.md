Collection of functions to help in the use of Keras.  

Recommended usage:  
```python
import messlkeras as mk
```  
Currently does not support `from messlkeras import *` .  

The following functions will be available in the module's namespace, ex: `mk.prep_data_SM(...)`

```python
prep_data_SpMa(spects_list, masks_list, input_shape=(100, 50, 513), start=0):
    ### prepares the data for Keras
    # keras_inputs will hold noisy spectrograms
    # keras_targets will hold desired masks
    # masks_list, spects_list should have the corresponding filenames in the same order
    # input_shape will define the shape of the data: (sample_num, input_length, features) (must all be positive)
    # start=n allows the user to start later in the lists    
    # the spectrogram data will be normalized to [-1..1]

prep_CHIME_lists(spects_dir, masks_dir, set_type='train'):
    # given two directories, creates two lists of all the files (with paths) that will be used by keras
    # (once fed to  pred_data_xyz)
    # this function ensure that the filenames match to ensure proper order when fitting model
    # set_type = {'train', 'dev', 'test'}
    # this function is meant to be used on CHiME data only
    # spects_dir point to the directory in which the sub-directories contain matlab .mat files of spectrograms
    # (masks_dir points to the masks)
    # Note: this function does not check if spects_dir contain actual spectrograms, just '.mat' files with reasonable path
```
