# prepare data for keras
# version that does no normalization, no dB conversion, only spliting into proper sizes
# usage: call this for each file_list, with the same numbers

import numpy as np
import os
import warnings
import scipy.io as sio
import time


def prep_data_for_keras(file_list, input_shape=(100, 50, 513), start=0):
    ### prepares the data for Keras, using CHIME3 data only!
    # file_list details with .mat files to load (created by prep_list_for_keras)
    # input_shape will define the shape of the data: (sample_num, input_length, features) (must all be positive)
    # start=n allows the user to start later in the lists    
    # returns a numpy.ndarray of shape input_shape

    # check if shape is incorrect 
    if any(s<=0 for s in input_shape):
        raise Exception("input_shape has negative values! Must all be strictly greater than 0.")
    if input_shape[2] != 513:
        raise Exception("feature number different than CHIME3 normal of 513! Exiting.")

    sample_num, input_length, features = input_shape
    
    finished = False
    num_proc_files = 0
    
    keras_data = None
    
    # safety valve for runtime 
    max_allowed = 180 #180s = 3 minutes
    start_time = time.clock()
    time_used = 0
    
    while not(finished):
                
        # check amount of time used, and exit loop with warning if time exceeds limit
        if time_used >= max_allowed:
            warnings.warn("Time limit exceeded, returning as is!")
            break
            
        # load next masks and spectrogram, if fail return what we have with warning
        try:            
            loaded_data = sio.loadmat(file_list[num_proc_files+start])['data'][0][0][0]
        except:
            warnings.warn("Not enough files, returning as is!")
            finished = True
            break
        
        # extract useful variables
        try:
            # normal case
            freq_num, frame_num, nb_channels = loaded_data.shape
            # for CHIME3 audio spectrograms/masks, should be (513, ~100:400, 6)  
            # freq_num will be input_dim for keras model
        except:
            # data is not the right shape
            raise Exception("Data is not the right shape! Exiting.")
        
        # check if data needs to be replicated
        if nb_channels == 6:
            # normal case of 6 noisy channels spectrograms/masks
            pass
        elif nb_channels == 2:
            # spectrograms from stereo files, currently mvdr-beamformfiles.
            # default: keep only the second one.
            # replicate 6 times
            loaded_data = loaded_data[:,:,1::2] #start at index one, progress by steps of size 2
            # duplicate for each 6 channels of chime3
            loaded_data = np.tile(loaded_data, (1, 1, 6))
            nb_channels = 6
        elif nb_channels == 1:
            # case with only one spectrogram/mask, duplicate it 6 times
            # duplicate for each 6 channels of chime3
            loaded_data = np.tile(loaded_data, (1, 1, 6))
            nb_channels = 6
        else:
            # ??? currently no cases like this, throw error
            raise Exception("Data is not the right shape! Exiting.")
        
        
        # swap axes feat<->chan
        loaded_data = loaded_data.swapaxes(0,2)
        
        # pad end of data to fit ('wrap' = add frames from the beginning)
        # this sort of works for input_length > frame_num but leads to some repetitions
        amount_to_pad = input_length - (frame_num % input_length)
        loaded_data = np.pad(loaded_data, ((0,0),(0,amount_to_pad),(0,0)), 'wrap')
       
        # reshape to keras desired shape
        # -1 allows for automatic dimension calculation
        temp_keras_data = loaded_data.reshape(-1, input_length, features)

        # add to arrays to return 
        if keras_data is None:
            keras_data = temp_keras_data
        else:
            keras_data = np.concatenate((keras_data, temp_keras_data), axis=0)
       
        # check if we are done
        if len(keras_data) >= sample_num: finished=True
        
        # increment the number of processed files
        num_proc_files += 1
        
        # update time_used
        time_used = time.clock() - start_time
        
    # return the files in the right format
    if keras_data is None:
        return (None, 0)
    else:
        return (keras_data[:sample_num], num_proc_files)
