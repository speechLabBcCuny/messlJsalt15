# Train an LSTM to take single channel noisy input spectrogram and produce a mask as output. 

import numpy as np
import os
import warnings
import scipy.io as sio
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.layers import LSTM, Dense, Convolution1D
import time
import random
import string

import messlkeras as mk

####################################################
# 1 - run an experiment, saving results on the way #
####################################################

# print experiment time for logging
print "Running experiment."
start_time = time.strftime('%Y-%m-%d %T')
print start_time

print "Building model: input->LSTM:1024->Dense:513=output :: optimizer=RMSprop,loss=binary_crossentropy"
# define sequential model
model = Sequential()
# the 1st LSTM layer
model.add(LSTM(input_dim=513, input_length=None, output_dim=1024, return_sequences=True))
# output layer
model.add(TimeDistributed(Dense(output_dim=513)))
model.compile(optimizer='RMSprop',loss='binary_crossentropy')

# data preparation
print "Preparing data."
work_dir = "/home/data/CHiME3/data/audio/16kHz/local/"
masks_dir = work_dir+"/messl-masks/ideal_amplitude/data/"
spects_dir = work_dir+"/messl-spects/data/"
save_dir = "/home/data/CHiME3/experiments/"

print "Working in directory:", work_dir
# build list of filenames (including paths)
print "Building masks files list from:", masks_dir
masks_list = \
    [path+'/'+file for path,_,files in sorted(os.walk(masks_dir)) for file in sorted(files) if file.endswith('.mat')]
print "Number of mask files is:", len(masks_list)

print "Building spects files list from:", spects_dir
spects_list = \
    [path+'/'+file for path,_,files in sorted(os.walk(spects_dir)) for file in sorted(files) if file.endswith('.mat')]
print "Number of spects files is:", len(spects_list)

# check if filenames match, otherwise throw error
print "Checking if masks/spects filenames match."
if [x.split('/')[-1] for x in masks_list] == [x.split('/')[-1] for x in spects_list]:
    print "Filenames match."
else:
    raise Exception("Filenames do not match!")

# create new experiment folder
print "Creating new folder for this experiment in:", save_dir
newexp_folder_path = save_dir+'/'+"exp_"+start_time+'/'
os.makedirs(newexp_folder_path)
    
# preliminaries
start_from_file = 0
seed = random.random()
print "Random seed is:", seed
random.seed(seed)
run = 0

# start continuous experiment
while run<100:
    
    print "Run number", run
        
    # prepare new samples for keras
    num_samples = 1000
    length_samples = random.randrange(10,200)
    print "Preparing data for Keras: num_samples={0}, length_samples={1}".format(num_samples, length_samples)
    keras_inputs, keras_targets, num_proc_files = mk.prep_data_SM(\
    	spects_list, masks_list, input_shape=(num_samples, length_samples, 513), start=start_from_file)
    
    # update next starting file
    start_from_file += num_proc_files
    # reset start_from_file if the end of the list has been reached
    if start_from_file>=len(masks_list): start_from_file=0
    
    # fit model
    nb_epoch=100
    batch_size=128
    print "Beginning fit: nb_epoch={0}, batch_size={1}".format(nb_epoch, batch_size)
    hist = model.fit(keras_inputs, keras_targets, nb_epoch, batch_size, verbose=2, shuffle=True)

    # save model to file every 1 run
    model_filename = "run_{0:04d}_keras_model.hdf5".format(run)
    print "saving model to {0}".format(model_filename)
    model.save(newexp_folder_path+model_filename)
    
    # save history to text file
    hist_filename = "run_{0:04d}_keras_history.txt".format(run)
    print "saving history to {0}".format(hist_filename)
    with open(newexp_folder_path+hist_filename, "w") as text_file:
        text_file.write("epoch: {}\n".format(hist.epoch))
        text_file.write("loss: {}\n".format(hist.history["loss"]))
    

    run +=1
