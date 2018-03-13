# prepare data for keras
# version that does no normalization, no dB conversion, only spliting into proper sizes
# usage: call this for each file_list, with the same numbers

import numpy as np
import os
import warnings
import scipy.io as sio
import time
import scipy as sp


def prep_data_for_keras(file_list, data_type, input_shape=(-1, 150, 513), start=0, time_limit=180, verbose=False):
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

    # for cases with 2 channels, keep the right one
    if data_type=='target_spect':
        # data is a MVDR spectrogram
        chan2keep = 1
    else:
        chan2keep = 0


    for filename in file_list[start:]:

        if verbose: print("working on", filename)

        # check amount of time used, and exit if time exceeds limit (will cause problem with other loaded data, no point returning anything)
        if time_used >= time_limit:
            raise Exception("Time limit exceeded!, Exiting")

        # load next .mat file, if fail exit (this should not happen if the file_list is well built)
        try:
            loaded_data = sio.loadmat(filename,verify_compressed_data_integrity=False)['data'][0][0][0]
        except:
            # faile to load file
            raise Exception("Error Loading file {}! Exiting.".format(filename))

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
            # then replicate 6 times
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

        # reshape to keras desired shape
        loaded_data_chans, loaded_data_length, _ = loaded_data.shape


        ### treat differently for each data case
        if data_type=='input_spect':
            # input spect
            # convert complex values to decibel
            loaded_data =  np.nan_to_num(np.log10(abs(loaded_data)))
            # normalize
            means = np.swapaxes(np.tile(loaded_data.mean(axis=1),(loaded_data_length,1,1)),0,1)
            stds = np.swapaxes(np.tile(loaded_data.std(axis=1),(loaded_data_length,1,1)),0,1)
            loaded_data = (loaded_data- means) / stds

        elif data_type=='input_mask':
            # input masks
            # apply logit
            loaded_data =  np.nan_to_num(sp.special.logit(loaded_data))

        elif data_type=='target_mask':
            # target mask
            pass

        elif data_type=='target_spect':
            # target spect
            # convert complex values to decibel
            loaded_data =  np.nan_to_num(np.log10(abs(loaded_data)))

        elif data_type=='theta':
            # data for which we need the np.angle
            loaded_data = np.angle(loaded_data)

        elif data_type=='abs':
            # complex data for which we need only the abs val
            loaded_data = abs(loaded_data)

        else:
            raise Exception("Unspecified data type! Exiting.")

        # add to arrays to return
        if keras_data is None:
            # pre-allocate memory of correct type & size, only once
            # fill with zeros, maximum size (real+simu tr05) ~16gb
            keras_data = np.zeros((12213449, features), dtype='float32')
            if verbose: print("Keras data created with size {}.".format(keras_data.nbytes))
            
        #     OLD
        # if keras_data is None:
        #     # pre-allocate memory of correct type & size, only once
        #     if sample_num < 0:
        #         # fill with zeros, maximum size (real+simu tr05)
        #         keras_data = np.zeros((1+2066520, features), dtype='float32')
        #         print("Keras data created with size {}.".format(keras_data.nbytes))
        #     else:
        #         keras_data = np.zeros((num_sample*), dtype='float32')
        #         print("Keras data created with size {}.".format(keras_data.nbytes))

        # flatten samples
        loaded_data_chans, loaded_data_length, _ = loaded_data.shape
        loaded_data = np.reshape(loaded_data, (loaded_data_chans*loaded_data_length, 513))
        
        
        # insert in proper position
        # if not enough, crash (desired behavior)
        keras_data[pos_to_insert:pos_to_insert+len(loaded_data)] = loaded_data
        pos_to_insert += len(loaded_data)


        # # insert in proper position
        # if (pos_to_insert + len(loaded_data)) < len(keras_data):
        #     # there is still room
        #     keras_data[pos_to_insert:pos_to_insert+len(loaded_data)] = loaded_data
        #     pos_to_insert += len(loaded_data)
        # else:
        #     # we are out of room
        #     room_left = len(keras_data[pos_to_insert:])
        #     keras_data[pos_to_insert:] = loaded_data[:room_left]
        #     pos_to_insert += len(loaded_data[:room_left])


        # increment the number of processed files
        num_proc_files += 1

        # update time_used
        time_used = time.clock() - start_time

        # check if we are done
        if sample_num>0 and pos_to_insert>=sample_num*input_length: break

    # done with files, ready to return
    
    # first, remove trailing zeros and make correct shape
    if verbose: print("removing {} zeros".format(pos_to_insert%input_length) )
    keras_data.resize((int(pos_to_insert/input_length), input_length, 513))
    if verbose: print("Kera_data resized with size {}".format(keras_data.nbytes))

    # if no sample_num specified
    if sample_num<0:
        #nothing to do
        pass
    elif sample_num>0 and (len(keras_data) > sample_num):
        # inserted slightly more than needed
        if verbose: print("removing {} samples".format(len(keras_data)-sample_num) )
        keras_data.resize((sample_num,input_length, 513))
        if verbose: print("Kera_data resized with size {}".format(keras_data.nbytes)) 

    elif len(keras_data) < sample_num:
        warnings.warn("All files loaded but too many samples asked, returning as is!")

    return (keras_data, num_proc_files)
