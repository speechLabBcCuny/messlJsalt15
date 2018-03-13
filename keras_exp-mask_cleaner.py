
# Continuously trains mask-cleaner model of increasing complexity
# input: noisy spectrogram + messl mask
# output: mask

# Does a naive (random) hyper parameter search

# models trained on ideal amplitude or phase sensitive ideal masks, or on spectrogram

### IMPORTS
from __future__ import print_function
import sys
import time
import random
import os

# import logging
# logging.getLogger('tensorflow').disabled = True

# # set verbosity
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'

import tensorflow as tf
# allow GPU growth
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.allow_soft_placement=True
config.log_device_placement=True
set_session(tf.Session(config=config))

# tf.logging.set_verbosity(tf.logging.FATAL)

import numpy as np

import json
import cPickle as pickle

import messlkeras as mk
from keras.models import Model

### PRELIMINARIES

# script arguments
# name of this script
script_name = sys.argv[0]
# where the new experiment results should be saved
save_dir = sys.argv[1] #string, directory
# desired GPU
gpu_num =  sys.argv[2]

#choose a GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
print("Running script on GPU", gpu_num)
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_num # desired GPU


# # set verbosity
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'

print("Running experiment from script: ", script_name)
start_time = time.strftime('%Y-%m-%d_%T')
print("Start Time: ", start_time)

print("save dir: ", save_dir)

# create new experiment folder using timestamp
print("Creating new folder for this experiment in:", save_dir)
newexp_folder_path = save_dir+'/'+start_time+'/'
os.makedirs(newexp_folder_path)


# lists of files to work on
data_dir = "/home/data/CHiME3/data/audio/16kHz/local/"
# messl_masks_dir = "/scratch/mim/chime3/messlMcMvdrMrf.2Hard5Lbp4Slate/data/" # hard masks
messl_masks_dir = "/scratch/proj/messl/vanilla_MESSL_output/MESSL_softmasks/" #soft masks

# prepare all the lists
# tr_lists, dt_lists, et_lists = mk.prep_chime3_lists() # use this is the files below don't work

# precomp_dir = "/scratch/felix/MESSLKERAS/pre-computed-data/"
# tr_lists = pickle.load( open( precomp_dir + '/tr_lists.pkl', "rb" ) )
# dt_lists = pickle.load( open( precomp_dir + '/dt_lists.pkl', "rb" ) )
# et_lists = pickle.load( open( precomp_dir + '/et_lists.pkl', "rb" ) )

# [chime_spects_noisy_tr, messl_soft_masks_tr, iaf_masks_tr, psf_masks_tr, messl_spects_mvdr_tr] = tr_lists
# [chime_spects_noisy_dt, messl_soft_masks_dt, iaf_masks_dt, psf_masks_dt, messl_spects_mvdr_dt] = dt_lists
# [chime_spects_noisy_et, messl_soft_masks_et, iaf_masks_et, psf_masks_et, messl_spects_mvdr_et] = et_lists # not used

# make smaller for debugging
# print("making data smaller for debugging")
# size = 1
# chime_spects_noisy_tr = chime_spects_noisy_tr[0:size]
# messl_soft_masks_tr = messl_soft_masks_tr[0:size]
# iaf_masks_tr = iaf_masks_tr[0:size]
# psf_masks_tr = psf_masks_tr[0:size]
# messl_spects_mvdr_tr = messl_spects_mvdr_tr[0:size]

# chime_spects_noisy_dt = chime_spects_noisy_dt[0:size]
# messl_soft_masks_dt = messl_soft_masks_dt[0:size]
# iaf_masks_dt = iaf_masks_dt[0:size]
# psf_masks_dt = psf_masks_dt[0:size]
# messl_spects_mvdr_dt = messl_spects_mvdr_dt[0:size]


# load the precomputed the data that is used by all experiments
# ie    chime_spects_noisy_tr messl_soft_masks_tr (logit)
# this data should already be normalized
precomp_dir = "/scratch/felix/MESSLKERAS/pre-computed-data/"

print("loading precomputed data from {}".format(precomp_dir))
load_start_time = time.strftime('%Y-%m-%d_%T')
print("Loading start Time: ", load_start_time)
keras_chime_spects_noisy_tr = np.load(precomp_dir + '/keras_chime_spects_noisy_tr.npy')
keras_messl_soft_masks_tr = np.load(precomp_dir + '/keras_messl_soft_masks_tr.npy')
load_end_time = time.strftime('%Y-%m-%d_%T')
print("Loading end Time: ", load_end_time)


#check that sizes match
if (keras_chime_spects_noisy_tr.shape == keras_messl_soft_masks_tr.shape):
    print("Shape of training data:", keras_chime_spects_noisy_tr.shape)
else:
    raise Exception("Training sample sizes do not match! Exiting.")

keras_chime_spects_noisy_dt = np.load(precomp_dir + '/keras_chime_spects_noisy_dt.npy')
keras_messl_soft_masks_dt = np.load(precomp_dir + '/keras_messl_soft_masks_dt.npy')

#check that sizes match
if (keras_chime_spects_noisy_dt.shape == keras_messl_soft_masks_dt.shape):
    print("Shape of validation data:", keras_chime_spects_noisy_dt.shape)
else:
    raise Exception("Validation sample sizes do not match! Exiting.")

sample_num, input_length, feat_num = keras_messl_soft_masks_tr.shape

# start rolling experiments

# starting complexities
num_layers = 3
layer_complexity = 3
roll = 8

# which experiments to test
exps_to_try = ['msa', 'iaf', 'psf', 'msf']

# never ends
while True:

    print("Starting roll {}.".format(roll))

    # get a new config of increasing complexity
    if roll<=3:
        # model config for rolls 0:3
        # here we test which bid merge mode is best
        layer_sizes=[1024,1024,1024]
        # layer_sizes=[10]
        bid_merge_mode=['sum', 'mul', 'concat', 'ave'][roll]
        out_activation='hard_sigmoid' # ['elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
        drop_rate=0.5
    elif roll<=7:
        # model config for rolls 4:7
        # here we test which activation function is the best
        layer_sizes=[1024,1024,1024]
        # layer_sizes=[10]
        bid_merge_mode='ave'
        out_activation= ['sigmoid', 'hard_sigmoid', 'tanh', 'softsign'][roll-4]
        drop_rate=0.5

    else:
        # model config afterwards
        # we keep adding layers and doubling layer size
            #  more geeky
            # slice_num = (np.sqrt(8*roll-1)-1)/2
            # slice_step = np.floor(slice_num-np.floor(slice_num))*np.floor(slice_num)
        # define layer_sizes, of increasing complexity
        layer_sizes = []
        for l in range(num_layers):
            layer_sizes.append(128*(2**layer_complexity))
        layer_complexity+=1
        if layer_complexity>num_layers:
            layer_complexity=0
            num_layers+=1
        # print(num_layers,layer_complexity)
        # print(layer_sizes
        bid_merge_mode ='ave'
        out_activation = 'hard_sigmoid'
        drop_rate=0.5

    # work on each exp type
    for exp_type in exps_to_try:

        # create new experiment folder using timestamp
        subexp_start_time = time.strftime('%Y-%m-%d_%T')
        subexp_folder_path = newexp_folder_path+'/'+exp_type+'_'+subexp_start_time+'/'
        print("Creating new folder for this experiment in:", subexp_folder_path)
        os.makedirs(subexp_folder_path)

        # load targets associated with exp_type
        keras_targets_tr_file = {'iaf':'/keras_iaf_masks_tr.npy', 'psf':'/keras_psf_masks_tr.npy', 'msa':'/keras_messl_spects_mvdr_tr.npy', 'psa':'/keras_psa_targets_tr.npy'}[exp_type]
        keras_targets_dt_file = {'iaf':'/keras_iaf_masks_dt.npy', 'psf':'/keras_psf_masks_dt.npy', 'msa':'/keras_messl_spects_mvdr_dt.npy', 'psa':'/keras_psa_targets_dt.npy'}[exp_type]

        print("loading targets associated with exp_type {}".format(exp_type))
        keras_targets_tr = np.load(precomp_dir+keras_targets_tr_file)
        keras_targets_dt = np.load(precomp_dir+keras_targets_dt_file)

        # # print("making data smaller for debugging")
        # print("making smaller for debugging")
        # size = 10
        # keras_chime_spects_noisy_tr = keras_chime_spects_noisy_tr[:size]
        # keras_messl_soft_masks_tr = keras_messl_soft_masks_tr[:size]
        # keras_targets_tr = keras_targets_tr[:size]
        # keras_chime_spects_noisy_dt = keras_chime_spects_noisy_dt[:size]
        # keras_messl_soft_masks_dt = keras_messl_soft_masks_dt[:size]
        # keras_targets_dt = keras_targets_dt[:size]


        # create model associated with exp_type
        print("Creating model with config:")
        print(config)
        config, model = mk.new_combo2mask_model(input_length=input_length,
                                                exp_type=exp_type, 
                                                layer_sizes=layer_sizes,
                                                bid_merge_mode=bid_merge_mode,
                                                out_activation=out_activation,
                                                drop_rate=drop_rate)

        ### train model

        # model training params
        optimizer='nadam'
        if exp_type in ['iaf', 'psf']:
            loss='binary_crossentropy'
        if exp_type in ['psa', 'msa']:
            loss='mean_squared_error'
        batch_size = 32
        epochs = 15
        patience = 5
        period = 1

        print("saving experiment parameters, config, model architecture")

        with open(subexp_folder_path+"/experiment_configuration.txt", "w") as config_txt:
            print("Description of the model experiment", file=config_txt)
            print("experiment type: {}".format(exp_type), file=config_txt)
            print("model config:", file=config_txt)
            print(config, file=config_txt)
            print("optimizer:", optimizer, file=config_txt)
            print("batch_size:", batch_size, file=config_txt)

        with open(subexp_folder_path+'/model_architecture.json', 'w') as outfile:
            # json.dump(mask_preds_model.to_json(), outfile)
            outfile.write(model.to_json())

        pickle.dump(config, 
                    open(subexp_folder_path + '/config.pkl', 'wb'), 
                    protocol=pickle.HIGHEST_PROTOCOL)

        # TRAIN!
        needs_training = True
        attempt = 0
        # while needs_training:
        #     try:
        tr_start_time = time.strftime('%Y-%m-%d_%T')
        print("Starting training attempt {}".format(attempt))
        print("Training start Time: ", tr_start_time)
        hist, model = mk.train_keras_model( model = model, 
                                            save_dir = subexp_folder_path,
                                            tr_inputs = [keras_chime_spects_noisy_tr, keras_messl_soft_masks_tr], 
                                            tr_targets = keras_targets_tr,
                                            val_inputs = [keras_chime_spects_noisy_dt, keras_messl_soft_masks_dt],
                                            val_targets = keras_targets_dt,
                                            optimizer=optimizer, loss=loss, batch_size=batch_size, epochs=epochs,
                                            patience=patience, period=period)
        tr_end_time = time.strftime('%Y-%m-%d_%T')
        print("Training end Time: ", tr_end_time)

        needs_training = False

            # except:
            #     print("Attempt failed, retrying.")
            #     attempt+=1

        # save results
        print("Saving trained model weights, training history")

        model.save_weights(subexp_folder_path+'/final_model_weights.hdf5')

        pickle.dump(hist.history, 
                    open(subexp_folder_path + '/train_val_history.pkl', 'wb'), 
                    protocol=pickle.HIGHEST_PROTOCOL)


        # # delete targets to save ram (does this even work?)
        # del(keras_targets_tr)
        # del(keras_targets_dt)

    # end of roll
    roll +=1

# EOF (should never hit)
print("End of script: ", script_name)



#### OLD


# # # if target is mvdr audio, change chan2keep
# # if exp_type in ['msa', 'psa']: chan2keep=1
# # keras_targets_tr, _ = mk.prep_data_for_keras(target_list_tr, input_shape, start, chan2keep, time_limit=10800)

# #check that sizes match
# if (keras_inputs_spect_tr.shape == keras_targets_tr.shape) \
#     and (keras_inputs_spect_tr.shape == keras_inputs_mask_tr.shape):
#     print("Shape of training data:", keras_inputs_spect_tr.shape)
# else:
#     raise Exception("Training sample sizes do not match! Exiting.")

# # ### prepare validation data
# # sample_num, input_length, feat_num = (-1,150,513)
# # input_shape = (sample_num, input_length, feat_num)
# # start = 0
# # chan2keep = 0
# # keras_inputs_spect_dt, _ = mk.prep_data_for_keras(input_spect_list_dt, input_shape, start, chan2keep, time_limit=10800)
# # keras_inputs_mask_dt, _ = mk.prep_data_for_keras(input_mask_list_dt, input_shape, start, chan2keep, time_limit=10800)
# # if exp_type in ['msa', 'psa']: chan2keep=1
# # keras_targets_dt, _ = mk.prep_data_for_keras(target_list_dt, input_shape, start, chan2keep, time_limit=10800)

# # #check that sizes match
# # if (keras_inputs_spect_dt.shape == keras_targets_dt.shape) \
# #     and (keras_inputs_spect_dt.shape == keras_inputs_mask_dt.shape):
# #     print("Shape of validation data:", keras_inputs_spect_dt.shape)
# # else:
# #     raise Exception("Validation sample sizes do not match! Exiting.")


# # remove complex values, unfortunately means making a new copy
# print("Removing complex values from data, adapting to experiment type.")

# # if phase sensitive spectrogram approximation, apply to targets
# # apply cos(theta) for PSA loss
# if exp_type == 'psa':
#     theta_y = np.angle(keras_inputs_spect_tr)
#     theta_s = np.angle(keras_targets_tr)
#     theta = theta_y - theta_s
#     keras_targets_tr = abs(keras_targets_tr)*np.cos(theta)
#     del(theta)

#     theta_y = np.angle(keras_inputs_spect_dt)
#     theta_s = np.angle(keras_targets_dt)
#     theta = theta_y - theta_s
#     keras_targets_dt = abs(keras_targets_dt)*np.cos(theta)
#     del(theta)

# # apply logit to messls masks (input only)
# keras_inputs_mask_tr =  np.nan_to_num(sp.special.logit(keras_inputs_mask_tr))
# keras_inputs_mask_dt =  np.nan_to_num(sp.special.logit(keras_inputs_mask_dt))

# # make sure spects are abs value
# keras_inputs_spect_tr = abs(keras_inputs_spect_tr)
# keras_inputs_spect_dt = abs(keras_inputs_spect_dt)
# if exp_type == 'msa': #'psa' already done above
#     keras_targets_tr = abs(keras_targets_tr)
#     keras_targets_dt = abs(keras_targets_dt)

# # convert scpects to db
# keras_inputs_spect_tr =  np.log(keras_inputs_spect_tr)
# keras_inputs_spect_dt =  np.log(keras_inputs_spect_dt)
# # if targest are spects
# if exp_type in ['msa', 'psa']:
#     keras_targets_tr =  np.log(keras_targets_tr)
#     keras_targets_dt=  np.log(keras_targets_dt)


# # run single training instance currently only iaf, psf

# # new model
# print("Running single training instance ")

# # model config
# layer_sizes=[512,512]
# bid_merge_mode="concat"
# out_activation="relu"
# drop_rate=0.5

# config, model = mk.new_combo2mask_model(input_length=input_length,
#                                         layer_sizes=layer_sizes,
#                                         bid_merge_mode=bid_merge_mode,
#                                         out_activation=out_activation,
#                                         drop_rate=drop_rate)


# # model training params
# optimizer='nadam'
# if exp_type in ['iaf', 'psf']:
#     loss='binary_crossentropy'
# if exp_type in ['psa', 'msa']:
#     loss='mean_squared_error'
# batch_size = 64
# epochs = 250

# tr_start_time = time.strftime('%Y-%m-%d_%T')
# print("Training start Time: ", tr_start_time)
# hist, model = mk.train_keras_model( model = model, 
#                                     save_dir = newexp_folder_path,
#                                     tr_inputs = [keras_inputs_spect_tr, keras_inputs_mask_tr], 
#                                     tr_targets = keras_targets_tr,
#                                     val_inputs = [keras_inputs_spect_dt, keras_inputs_mask_dt],
#                                     val_targets = keras_targets_dt,
#                                     optimizer=optimizer, loss=loss, batch_size=batch_size, epochs=epochs)

# tr_end_time = time.strftime('%Y-%m-%d_%T')
# print("Training end Time: ", tr_end_time)
# # save experiment config

# print("Training finished.")
# print("Saving experiment config, model weights, architecture, training history")

# with open(newexp_folder_path+"experiment_configuration.txt", "w") as config_txt:
#     print("Description of the model experiment", file=config_txt)
#     print("experiment type: {}".format(exp_type))
#     print("model config:", file=config_txt)
#     print(config, file=config_txt)
#     print("optimizer:", optimizer, file=config_txt)
#     print("batch_size:", batch_size, file=config_txt)

# pickle.dump(config, 
#             open(newexp_folder_path + 'config.pkl', 'wb'), 
#             protocol=pickle.HIGHEST_PROTOCOL)


# # save trained model
# model.save_weights(newexp_folder_path+'final_model_weights.hdf5')
# with open(newexp_folder_path+'model_architecture.json', 'w') as outfile:
#     json.dump(model.to_json(), outfile)

# pickle.dump(hist.history, 
#             open(newexp_folder_path + 'train_val_history.pkl', 'wb'), 
#             protocol=pickle.HIGHEST_PROTOCOL)


# # EOF
# print("End of script: ", script_name)
