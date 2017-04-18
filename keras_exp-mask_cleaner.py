
# Trains a nspect2mask model
# input: noisy spectrogram + messl mask
# output: mask

# Does a naive (random) hyper parameter search

# can be trained on ideal amplitude or phase sensitive ideal masks, or on spectrogram

# GOALS: Have this only save nspect2masks models

### IMPORTS
from __future__ import print_function
import sys
import time
import random

from keras.models import Sequential
from keras.layers import LSTM, Dense, Lambda, Merge
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.backend.tensorflow_backend import set_session
from keras import backend as K

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.allow_soft_placement=True
config.log_device_placement=True
set_session(tf.Session(config=config))
tf.logging.set_verbosity(tf.logging.ERROR)

import numpy as np
import scipy as sp

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="1" # desired GPU

import messlkeras as mk

### PRELIMINARIES

# script arguments
# name of this script
script_name = sys.argv[0]
# where the new experiment results should be saved
save_dir = sys.argv[1] #string, directory
# should we train on ideal amplitude or phase sensitive masks, on the cleaned psectrogram (phase sensitive or not)
exp_type = sys.argv[2] # string, shoud be either 'iaf' or 'psf' or 'msa' or 'psa'


print("Running experiment from script: ", script_name)
start_time = time.strftime('%Y-%m-%d %T')
print("Start Time: ", start_time)

print("Config: ")
print("save dir: ", save_dir)
print("exp type: ", exp_type)


# create new experiment folder using timestamp
print("Creating new folder for this experiment in:", save_dir)
newexp_folder_path = save_dir+'/'+"exp_"+start_time+'/'
os.makedirs(newexp_folder_path)


# lists of files to work on
data_dir = "/home/data/CHiME3/data/audio/16kHz/local/"
messl_masks_dir = "/scratch/mim/chime3/messlMcMvdrMrf.2Hard5Lbp4Slate/data/"

# training data
input_spect_list_tr = mk.prep_list_for_keras(data_dir, "messl-spects-noisy.*tr05.*real", verbose=False)
input_masks_list_tr = mk.prep_list_for_keras(messl_masks_dir, ".*tr05.*real", verbose=False)


if exp_type == 'iaf':
    target_list_tr = mk.prep_list_for_keras(data_dir, "mask.*ideal_amplitude.*tr05.*real", verbose=False)
elif exp_type == 'psf':
    target_list_tr = mk.prep_list_for_keras(data_dir, "mask.*phase_sensitive.*tr05.*real", verbose=False)
elif exp_type in ['msa', 'psa']:
    target_list_tr = mk.prep_list_for_keras(data_dir, "messl-spects-mvdr-cleaned.*tr05.*real", verbose=False)

# check that they match:
if ([x.split('/')[-1] for x in input_spect_list_tr] == [x.split('/')[-1] for x in target_list_tr]) \
    and ([x.split('/')[-1] for x in input_spect_list_tr] == [x.split('/')[-1] for x in input_masks_list_tr]):
    print("Number of training files:", len(input_spect_list_tr))
else:
    raise Exception("Training Filenames do not match! Exiting")

# validation data (from dt05)
# training data
input_spect_list_dt = mk.prep_list_for_keras(data_dir, "messl-spects-noisy.*dt05.*real", verbose=False)
input_masks_list_dt = mk.prep_list_for_keras(messl_masks_dir, ".*dt05.*real", verbose=False)

if exp_type == 'iaf':
    target_list_dt = mk.prep_list_for_keras(data_dir, "mask.*ideal_amplitude.*dt05.*real", verbose=False)
elif exp_type == 'psf':
    target_list_dt = mk.prep_list_for_keras(data_dir, "mask.*phase_sensitive.*dt05.*real", verbose=False)
elif exp_type in ['msa', 'psa']:
    target_list_dt = mk.prep_list_for_keras(data_dir, "messl-spects-mvdr-cleaned.*dt05.*real", verbose=False)

# check that they match:
if ([x.split('/')[-1] for x in input_spect_list_dt] == [x.split('/')[-1] for x in target_list_dt]) \
    and ([x.split('/')[-1] for x in input_spect_list_dt] == [x.split('/')[-1] for x in input_masks_list_dt]):
    print("Number of validation files:", len(input_spect_list_dt))
else:
    raise Exception("Training Filenames do not match! Exiting")


# make smaller for debugging
size = 1
input_spect_list_tr = input_spect_list_tr[0:size]
input_masks_list_tr = input_masks_list_tr[0:size]
target_list_tr = target_list_tr[0:size]

input_spect_list_dt = input_spect_list_dt[0:size]
input_masks_list_dt = input_masks_list_dt[0:size]
target_list_dt = target_list_dt[0:size]

### prepare data
sample_num, input_length, feat_num = (-1,20,513)
input_shape = (sample_num, input_length, feat_num)
start = 0
chan2keep = 0

keras_inputs_spect_tr, _ = mk.prep_data_for_keras(input_spect_list_tr, input_shape, start, chan2keep, time_limit=7200)
keras_inputs_masks_tr, _ = mk.prep_data_for_keras(input_masks_list_tr, input_shape, start, chan2keep, time_limit=7200)

keras_targets_tr, _ = mk.prep_data_for_keras(target_list_tr, input_shape, start, chan2keep, time_limit=7200)

#check that sizes match
if (keras_inputs_spect_tr.shape == keras_targets_tr.shape) \
    and (keras_inputs_spect_tr.shape == keras_inputs_masks_tr.shape):
    print("Shape of training data:", keras_inputs_spect_tr.shape)
else:
    raise Exception("Sample sizes do not match! Exiting.")


keras_inputs_spect_dt, _ = mk.prep_data_for_keras(input_spect_list_dt, input_shape, start, chan2keep, time_limit=7200)
keras_inputs_masks_dt, _ = mk.prep_data_for_keras(input_masks_list_dt, input_shape, start, chan2keep, time_limit=7200)

keras_targets_dt, _ = mk.prep_data_for_keras(target_list_dt, input_shape, start, chan2keep, time_limit=7200)

#check that sizes match
if (keras_inputs_spect_dt.shape == keras_targets_dt.shape) \
    and (keras_inputs_spect_dt.shape == keras_inputs_masks_dt.shape):
    print("Shape of validation data:", keras_inputs_spect_dt.shape)
else:
    raise Exception("Sample sizes do not match! Exiting.")

# if phase sensitive spectrogram approximation, apply to targets
# apply cos(theta) for PSA loss
if exp_type == 'psa':
    theta_y = np.angle(keras_inputs_spect_tr)
    theta_s = np.angle(keras_targets_tr)
    theta = theta_y - theta_s
    keras_targets_tr = abs(keras_targets_tr)*np.cos(theta)

    theta_y = np.angle(keras_inputs_spect_dt)
    theta_s = np.angle(keras_targets_dt)
    theta = theta_y - theta_s
    keras_targets_dt = abs(keras_targets_dt)*np.cos(theta)

# apply logit to messls masks (input only)
keras_inputs_masks_tr =  np.nan_to_num(sp.special.logit(keras_inputs_masks_tr))
keras_inputs_masks_dt =  np.nan_to_num(sp.special.logit(keras_inputs_masks_dt))


# make inputs proper size if needed xxx
if exp_type in ['msa', 'psa']:
    keras_inputs_tr = [keras_inputs_spect_tr,keras_inputs_spect_tr,keras_inputs_masks_tr]
    keras_inputs_dt = [keras_inputs_spect_dt,keras_inputs_spect_dt,keras_inputs_masks_dt]
elif exp_type in ['iaf', 'psf']:
    keras_inputs_tr = [keras_inputs_spect_tr,keras_inputs_masks_tr]
    keras_inputs_dt = [keras_inputs_spect_dt,keras_inputs_masks_dt]

def new_random_combo2mask_model():
    ## 1 - create the LSTM model: spect -> mask

    # parameters of the lstm
    feat_num = 513
    output_dim = 513

    batch_norm_mode = random.choice([0,2])
    numlayers = random.choice([1,2])
    layersizes = []
    for layer in range(numlayers):
        layersizes.append( random.choice([256,512,1024,2048]) )
    bid_merge_mode = random.choice(['sum', 'mul', 'concat', 'ave'])
    activation = random.choice(['sigmoid', 'hard_sigmoid'])
#     optimizer = random.choice(['SGD', 'RMSprop', 'Adam', 'Adamax', 'Nadam'])
    optimizer = 'RMSprop'
    config = [batch_norm_mode, numlayers, layersizes, bid_merge_mode, activation, optimizer] 
    # Hard code config if needed
    # [batch_norm_mode, numlayers, layersizes, bid_merge_mode, activation, optimizer] = 
    print("config is [batch_norm_mode, numlayers, layersizes, bid_merge_mode, activation, optimizer]: ", config )

    # spect sequential model
    model_nspect = Sequential()
    # conversion to dB  #
    model_nspect.add(TimeDistributed(Lambda(lambda x: K.log(K.abs(x))), input_shape=(None,feat_num))) 
    # normalize per feature per batch
    model_nspect.add(BatchNormalization(mode=batch_norm_mode, axis=2, input_shape=(None,513)))

    # MESSL mask seq model
    model_mmask = Sequential()
    # normalize per feature per batch
    model_mmask.add(BatchNormalization(mode=batch_norm_mode, axis=2, input_shape=(None,513)))

    # merge mmask with spect
    model_combo2mask = Sequential()
    model_combo2mask.add(Merge([model_nspect,model_mmask], mode='concat'))

    # bidirectional lstm layer(s)
    for i in range(numlayers):
        model_combo2mask.add(Bidirectional(LSTM(input_length=None, output_dim=layersizes[i], return_sequences=True), merge_mode=bid_merge_mode ))

    # out layer
    model_combo2mask.add(TimeDistributed(Dense(output_dim=output_dim, activation=activation )))

    # for a mean squared error regression problem
    model_combo2mask.compile(optimizer=optimizer, loss='binary_crossentropy')

    return [config, model_combo2mask]


# Run trials
for trial_num in range(100)[1:]:
    print("New Trial number ", trial_num)

    # callbacks
    early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, verbose=0, mode='auto')

    # model_checker = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=3)
    
    callbacks = [early_stopper]

    config, model_combo2mask = new_random_combo2mask_model()

    if exp_type in ['iaf', 'psf']:
        model_to_train = model_combo2mask
    elif exp_type in ['msa', 'psa']:
        # spect sequential model 2
        model_nspect2 = Sequential()
        # conversion to dB  #
        model_nspect2.add(TimeDistributed(Lambda(lambda x: K.abs(x)), input_shape=(None,feat_num))) 
        
        model_to_train = Sequential()
        model_to_train.add(Merge([model_nspect2, model_combo2mask], mode='mul'))
   
        # for a mean squared error regression problem
        model_to_train.compile(optimizer='RMSprop', loss='mse')


    hist = model_to_train.fit(keras_inputs_tr, keras_targets_tr, batch_size=64, nb_epoch=100, shuffle=True, verbose=2, validation_data=(keras_inputs_dt, keras_targets_dt), callbacks = callbacks)

    print("Trial done.")
    cur_time = time.strftime('%Y-%m-%d %T')
    print(cur_time)

    exp_folder_path = newexp_folder_path
    val_loss = hist.history['val_loss'][-1]
    filepath = exp_folder_path + "/" + exp_type + "_model_"+cur_time+"_vl:"+str(val_loss)+".hdf5"
    print("Saving combo2mask model: ", filepath)
    model_combo2mask.save(filepath)

    print("Saving config of model")
    with open(exp_folder_path + "/" + cur_time + "_config.txt", "w") as config_file:
        config_file.write(str(config))
