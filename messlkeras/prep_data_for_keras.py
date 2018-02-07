# prepare data for keras
# version that does no normalization, no dB conversion, only spliting into proper sizes
# usage: call this for each file_list, with the same numbers

import numpy as np
import os
import warnings
import scipy.io as sio
import time


def prep_data_for_keras(file_list, input_shape=(-1, 50, 513), start=0, chan2keep='nan', time_limit=180, verbose=False):
    ### prepares the data for Keras, using CHIME3 data only!
    # file_list details with .mat files to load (created by prep_list_for_keras)
    # input_shape will define the shape of the data: (sample_num, input_length, features) (must all be positive)
    # input_shape[0]=-1 means that b default it will load all the files in the file_list, time permitting
    # start=n allows the user to start later in the lists
    # time_limit puts on cap on how long the process should take, in seconds. default is 3 minutes
    # chan2keep: in cases where the data is 2 channels, needs to specify which to keep. Should have value 0 or 1. Not required in other cases.

    # returns a numpy.ndarray of shape input_shape

    # check if shape is incorrect
    sample_num, input_length, features = input_shape

    if input_length<=0:
        raise Exception("Input length must be positive")
    if input_shape[2] != 513:
        raise Exception("feature number different than CHIME3 normal of 513! Exiting.")

    pos_to_insert = 0

    num_proc_files = 0

    keras_data = None

    start_time = time.clock()
    time_used = 0

    for filename in file_list[start:]:

        if verbose: print("working on", filename)

        # check amount of time used, and exit if time exceeds limit (will cause problem with other loaded data, no point continuing)
        if time_used >= time_limit:
            raise Exception("Time limit exceeded!, Exiting")

        # load next .mat file, if fail exit (this should not happen if the file_list is well built)
        try:
            loaded_data = sio.loadmat(filename)['data'][0][0][0]
        except:
            raise Exception("Error Loading file! Exiting.")

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
            # spectrograms from stereo files, currently mvdr-beamformfiles. keep chan 1 (ie the second channel)
            # OR masks from messls. keep chan 0
            # replicate 6 times
            if chan2keep not in [0,1]:
                raise Exception("2 channels found in .mat file. Please Specify chan2keep = 0 or 1. Exiting")
            else:
                loaded_data = loaded_data[:,:,chan2keep::2] #start at index one, progress by steps of size 2
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
        if frame_num >= input_length:
            amount_to_pad = frame_num % input_length
        else:
            amount_to_pad = input_length % frame_num

        loaded_data = np.pad(loaded_data, ((0,0),(0,amount_to_pad),(0,0)), 'wrap')

        # reshape to keras desired shape
        # -1 allows for automatic dimension calculation
        loaded_data = loaded_data.reshape(-1, input_length, features)


        # add to arrays to return
        if keras_data is None:
            # pre-allocate memory of correct type, only once
            if sample_num < 0:
                # fill with zeros, maximum size (real+simu tr05)
                keras_data = np.zeros((1+2066520/input_length, input_length, features), dtype=loaded_data.dtype)
            else:
                keras_data = np.zeros(input_shape, dtype=loaded_data.dtype)

        # insert in proper position
        keras_data[pos_to_insert:pos_to_insert+len(loaded_data)] = loaded_data
        pos_to_insert += len(loaded_data)

        # increment the number of processed files
        num_proc_files += 1

        # update time_used
        time_used = time.clock() - start_time

        # check if we are done
        if sample_num>0 and len(keras_data) >= sample_num: break


    # first, remove trailing zeros
    keras_data.resize((pos_to_insert,input_length,features))
    # return requested number of samples
    if sample_num>0 and len(keras_data) >= sample_num:
        keras_data.resize((sample_num,input_length,features))
    if len(keras_data) < sample_num:
        warnings.warn("All files loaded but too many samples asked, returning as is!")

    return (keras_data, num_proc_files)
