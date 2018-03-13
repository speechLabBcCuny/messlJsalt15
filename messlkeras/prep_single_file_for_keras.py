import scipy.io as sio
import scipy as sp
import numpy as np

def prep_single_file_for_keras(filename, data_type, verbose=False):
    ### prepares a single file for Keras, using CHIME3 data only!
    # filename is a .mat files to load
    # if the data is a spectrogram, the values are return as absolute value (soon:and converted to DB)
    # if the data is a input mask, logit is applied.
    # chan2keep: in cases where the data is 2 channels, needs to specify which to keep. Should have value 0 or 1. Not required in other cases.
    # returns a numpy.ndarray of shape (6, file utterance length, 513)



    # for cases with 2 channels, keep the right one
    if data_type=='target_spect':
        # data is a MVDR spectrogram
        chan2keep = 1
    else:
        chan2keep = 0


    if verbose: print("working on", filename)

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



    return loaded_data
