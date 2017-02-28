
# coding: utf-8

#######

# model that takes a spect+mask to predit a mask
# i.e. a mask cleaner

#######

# add messlKeras
import sys
import messlkeras as mk

import warnings
warnings.filterwarnings('always')

# imports
import math
import numpy as np

from keras.models import Sequential
from keras.layers import LSTM, Dense, Lambda, Merge
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed, Bidirectional

from keras.callbacks import EarlyStopping

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))

import keras

print "Tensorflow version ", tf.__version__
print "Keras version ", keras.__version__

import time
import random
import string
import os

import sys
import messlkeras as mk

import simplejson

#######

# print experiment time for logging
print "Starting preliminaries."
prelim_time = time.strftime('%Y-%m-%d_%T')
print prelim_time

#######

# create new experiment folder
save_dir = "/scratch/felix/IS2017-CHIME3-Exps/exp-LSTM-Mask-Cleaner/"
print "Creating new folder for this experiment in:", save_dir
newexp_folder_path = save_dir+'/'+"exp_"+prelim_time+'/'
os.makedirs(newexp_folder_path)

# lists of files to work on
print "Preparing data list."
data_dir = "/home/data/CHiME3/data/audio/16kHz/local/"
messl_masks_dir = "/scratch/mim/chime3/messlMcMvdrMrf.2Hard5Lbp4Slate/data/"

# training data
noisy_spects_list_tr = mk.prep_list_for_keras(data_dir, "messl-spects-noisy.*tr05.*real")
noisy_masks_list_tr = mk.prep_list_for_keras(messl_masks_dir, ".*tr05.*real")
target_masks_list_tr = mk.prep_list_for_keras(data_dir, "mask.*ideal_amplitude.*tr05.*real")
clean_spects_list_tr = mk.prep_list_for_keras(data_dir, "messl-spects-mvdr-cleaned.*tr05.*real")

# check that they match:
if [x.split('/')[-1] for x in noisy_spects_list_tr]\
	== [x.split('/')[-1] for x in clean_spects_list_tr]\
    == [x.split('/')[-1] for x in noisy_masks_list_tr]\
	== [x.split('/')[-1] for x in target_masks_list_tr]:
    print "Training Filenames match! Number of files: {}".format(len(noisy_spects_list_tr))
else:
    raise Exception("Training Filenames do not match! Exiting")

#######

# validation data (from dt05)
noisy_spects_list_dt = mk.prep_list_for_keras(data_dir, "messl-spects-noisy.*dt05.*real")
noisy_masks_list_dt = mk.prep_list_for_keras(messl_masks_dir, ".*dt05.*real")
#remove extra file
noisy_masks_list_dt = np.array([x for x in noisy_masks_list_dt if not(x.endswith('_2.mat'))])

target_masks_list_dt = mk.prep_list_for_keras(data_dir, "mask.*ideal_amplitude.*dt05.*real")
clean_spects_list_dt = mk.prep_list_for_keras(data_dir, "messl-spects-mvdr-cleaned.*dt05.*real")

# check that they match:
if [x.split('/')[-1] for x in noisy_spects_list_dt]\
    == [x.split('/')[-1] for x in clean_spects_list_dt]\
    == [x.split('/')[-1] for x in noisy_masks_list_dt]\
	== [x.split('/')[-1] for x in target_masks_list_dt]:
    print "Validating Filenames match! Number of files: {}".format(len(noisy_spects_list_dt))
else:
    raise Exception("Validating Filenames do not match! Exiting")


#######

### prepare data

sample_num, input_length, feat_num = (-1,150,513)
input_shape = (sample_num, input_length, feat_num)
start = 0

# prep training data
keras_m1_inputs_nspects_tr, num_proc_files1  = mk.prep_data_for_keras(noisy_spects_list_tr, input_shape, start, time_limit=7200)
keras_m1_inputs_masks_tr, num_proc_files1  = mk.prep_data_for_keras(noisy_masks_list_tr, input_shape, start, time_limit=7200)
keras_m1_targets_masks_tr, num_proc_files2 = mk.prep_data_for_keras(target_masks_list_tr, input_shape, start, time_limit=7200)
# check that they match:
if keras_m1_inputs_nspects_tr.shape\
	== keras_m1_inputs_masks_tr.shape\
	== keras_m1_targets_masks_tr.shape:
    print "Sample sizes match! Size is (num_samples, length, freqs)={}".format(keras_m1_inputs_nspects_tr.shape)
else:
    raise Exception("Sample sizes do not match! Exiting.")

# prep early stopping data (validating data from dt05)
keras_m1_inputs_nspects_dt, num_proc_files1  = mk.prep_data_for_keras(noisy_spects_list_dt, input_shape, start, time_limit=7200)
keras_m1_inputs_masks_dt, num_proc_files1  = mk.prep_data_for_keras(noisy_masks_list_dt, input_shape, start, time_limit=7200)
keras_m1_targets_masks_dt, num_proc_files2 = mk.prep_data_for_keras(target_masks_list_dt, input_shape, start, time_limit=7200)
# check that they match:
if keras_m1_inputs_nspects_dt.shape\
	== keras_m1_inputs_masks_dt.shape\
	== keras_m1_targets_masks_dt.shape:
    print "Sample sizes match! Size is (num_samples, length, freqs)={}".format(keras_m1_inputs_nspects_dt.shape)
else:
    raise Exception("Sample sizes do not match! Exiting.")



#######

# # apply cos(theta) for PSA loss
# theta_y = np.angle(keras_m1_inputs)
# theta_s = np.angle(keras_m2_targets)
# theta = theta_y - theta_s
# keras_m2_targets = np.abs(keras_m2_targets)*np.cos(theta)

#######

# apply logit to messls masks
keras_m1_inputs_masks_tr = np.array([max(10**-50, min(1-10**-50, x)) for x in keras_m1_inputs_masks_tr.flatten()])
keras_m1_inputs_masks_tr = np.array([math.log(x) - math.log(1-x) for x in keras_m1_inputs_masks_tr.flatten()])
keras_m1_inputs_masks_tr = keras_m1_inputs_masks_tr.reshape(input_shape)
keras_m1_inputs_masks_dt = np.array([max(10**-50, min(1-10**-50, x)) for x in keras_m1_inputs_masks_dt.flatten()])
keras_m1_inputs_masks_dt = np.array([math.log(x) - math.log(1-x) for x in keras_m1_inputs_masks_dt.flatten()])
keras_m1_inputs_masks_dt = keras_m1_inputs_masks_dt.reshape(input_shape)


#######

# name the data
X_train = [abs(keras_m1_inputs_nspects_tr), keras_m1_inputs_masks_tr] # [nspect, mmasks]
Y_train = keras_m1_targets_masks_tr

X_validate = [abs(keras_m1_inputs_nspects_dt), keras_m1_inputs_masks_dt]
Y_validate = keras_m1_targets_masks_dt


#######

## 1 - create the LSTM model: spect+mask -> mask
# merge spect with mask
# parameters of the lstm
# sample_num, input_length, feat_num = sample_num, input_length, feat_num // doesn't do anything just a reminder
output_dim = 513

# spect sequential model
model_nspect = Sequential()
# conversion to dB  # f = lambda x: tf.log(tf.abs(x))
model_nspect.add(TimeDistributed(Lambda(lambda x: tf.log(tf.abs(x))), input_shape = (None,feat_num))) 
# normalize per feature per batch
model_nspect.add(BatchNormalization(mode=0, axis=-1, input_shape = (None,513)))

# MESSL mask seq model
model_mmask = Sequential()
# normalize per feature per batch
model_mmask.add(BatchNormalization(mode=0, axis=-1, input_shape = (None,513)))

# merge mmask with spect
model_combo2mask = Sequential()
model_combo2mask.add(Merge([model_nspect,model_mmask], mode='concat'))

# bidirectional lstm layer
model_combo2mask.add(Bidirectional(LSTM(input_length=None, output_dim=1024, return_sequences=True)))

# out layer
model_combo2mask.add(TimeDistributed(Dense(output_dim=output_dim, activation='sigmoid')))
# for a mean squared error regression problem
model_combo2mask.compile(optimizer='RMSprop',loss='mse')

# print experiment time for logging
print "Preliminaries over. Start time of experiment part 1:"
start_time1 = time.strftime('%Y-%m-%d %T')
print start_time1

#######

# fit using early stopping

callbacks = [EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1, mode='auto')]
nb_epoch=100
batch_size=128
print "Part 1: pre-train mask recognizer, nb_epoch={0}, batch_size={1}.".format(nb_epoch,batch_size)
hist1= model_combo2mask.fit(X_train, Y_train, nb_epoch=nb_epoch, batch_size=batch_size, verbose=2, shuffle=True, validation_data=(X_validate, Y_validate), callbacks = callbacks)

# print experiment time for logging
print "Experiment part 1 over:"
end_time1 = time.strftime('%Y-%m-%d %T')
print end_time1

#######

# save model to file
model_filename = "keras_model_combo2mask.json"

# serialize model to JSON
model_combo2mask_json = model_combo2mask.to_json()
with open(newexp_folder_path+model_filename, "w") as json_file:
    json_file.write(simplejson.dumps(simplejson.loads(model_combo2mask_json), indent=4))

# serialize weights to HDF5
weights_filename = "keras_model_combo2mask_weights.h5"
model_combo2mask.save_weights(newexp_folder_path+weights_filename)
print("Saved model json and weights to disk")

# save history to text file
hist_filename = "keras_model_combo2mask_history.txt"
print "saving model1 history to {0}".format(hist_filename)
with open(newexp_folder_path+hist_filename, "w") as text_file:
    text_file.write("epoch: {}\n".format(hist1.epoch))
    text_file.write("loss: {}\n".format(hist1.history["loss"]))


print "End of script"





