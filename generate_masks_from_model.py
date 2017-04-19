import messlkeras as mk

import scipy.io as sio
import numpy as np

from keras.models import Sequential,load_model
from keras.layers import LSTM, Dense, Lambda, Merge
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed, Bidirectional
import keras.backend as K

import os
import sys

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))

model_path = sys.argv[1]
model_dir, model_name = os.path.split(model_path)

results_dir = model_dir + '/' + model_name[:-5]+"-results/"

# create directory in which well save results
save_dir = results_dir +'/masks/'
if not os.path.exists(save_dir):
    print("Creating directory with the masks from model in:", save_dir)
    os.mkdir(save_dir)

# is this a mask-cleaner of nspect2mask model?
if 'nspect2mask' in model_dir:
    model_type = 'nspect2mask'
elif 'mask_cleaner' in model_dir:
    model_type = 'mask_cleaner'
else:
    raise Exception("model type not recognized, exiting")

#load model
print("Loading model")
loaded_model = load_model(model_path)

# load and prepare input data
data_dir = '/home/data/CHiME3/data/audio/16kHz/local/messl-spects-noisy/data/'
messl_masks_dir = '/scratch/mim/chime3/messlMcMvdrMrf.2Hard5Lbp4Slate/data/'

if model_type == 'nspect2mask':
    input_spects_list = mk.prep_list_for_keras(data_dir, '[ed]t05')
elif model_type == 'mask_cleaner':
    input_spects_list = mk.prep_list_for_keras(data_dir, '[ed]t05')
    input_masks_list = mk.prep_list_for_keras(messl_masks_dir, '[ed]t05')

    # check that they match
    if [x.split('/')[-1] for x in input_spects_list] == [x.split('/')[-1] for x in input_masks_list]:
        print("number of files:", len(input_masks_list))
    else:
        raise Exception("Noisy spectrogram and Messl mask files do not match! Exiting")

# start predicting masks
for file_num in range(len(input_spects_list)):
    print("Working on file:", input_spects_list[file_num])

    noisy_spect = sio.loadmat(input_spects_list[file_num])['data'][0][0][0]
    noisy_spect = noisy_spect.swapaxes(0,2)
    noisy_spect = abs(noisy_spect)

    if model_type == 'nspect2mask':
        keras_input = noisy_spect
    if model_type == 'mask_cleaner':
        messl_mask = sio.loadmat(input_masks_list[file_num])['data'][0][0][0]
        messl_mask = messl_mask[:,:,0] #chan2keep from prep_data_for_keras.py
        messl_mask = np.tile(messl_mask, (1,1,6))
        messl_mask = swapaxes(0,2)

        keras_input = [noisy_spect, messl_mask]

    # predict mask
    predicted_mask = loaded_model.predict(keras_input)

    # apply min/max/ave
    mask = np.max(predicted_mask,axis=0)

    #save the mask to mat file
    save_file = save_dir + input_spects_list[file_num].replace(data_dir, "")
    # copy structure
    sub_save_dir = os.path.split(save_file)[0]
    if not os.path.exists(sub_save_dir):
        os.mkdir(sub_save_dir)

    sio.savemat(save_file, {'mask': mask})
    print("Mask saved")

print "finish"



# base_dir = '/scratch/near/test/mask/'
# #model_dir = '/scratch/felix/IS2017-CHIME3-Exps/exp-nspect2mask/psf/exp_2017-04-07 22:43:32/'
# #save_dir = '/scratch/near/CHiME3/fixed_model/v2/lstm-out-0412:15:42-min/'
# data_dir = '/home/data/CHiME3/data/audio/16kHz/local/messl-spects-noisy/data/'
# test_list = mk.prep_list_for_keras(data_dir, '[ed]t05')
# #weights_filename = "psf_model_2017-04-11 21:36:49_vl:0.164533952669.hdf5"

# print len(test_list)
# #load the latest model

# print "loading model...."
# # sequential model
# model = Sequential()
# # parameters of the lstm
# # sample_num, input_length, feat_num = sample_num, input_length, feat_num // doesn't do anything just a reminder
# output_dim = 513
# feat_num = 513
# # conversion to dB  # f = lambda x: tf.log(tf.abs(x))
# #model.add(TimeDistributed(Lambda(lambda x: tf.log(tf.abs(x))), input_shape = (None,feat_num)))

# # normalize per feature per batch
# #model.add(BatchNormalization(mode=0, axis=-1, input_shape = (None,513)))

# # bidirectional lstm layer
# #model.add(Bidirectional(LSTM(input_length=None, output_dim=1024, return_sequences=True)))

# # out layer
# #model.add(TimeDistributed(Dense(output_dim=output_dim, activation='sigmoid')))
# # for a mean squared error regression problem
# #model.compile(optimizer='RMSprop',loss='mse')
# # load weights into new model
# model=load_model(model_dir+'/'+weights_filename)
# print("Loaded model weights from disk")
# print "model loaded"
# print "......................"
# print "start predicting...."
# for test_file in test_list:
#     noisy = sio.loadmat(test_file)['data'][0][0][0]
#     noisy = noisy.swapaxes(0,2)
#     if model_type == ''
#     test_input = abs(noisy)
#     test_output = model.predict(test_input)
#     #combine the six masks to one single mask by taking the average.
#     mask = np.min(test_output,axis=0)
#     #save the mask to mat file
#     save_file = save_dir+test_file.replace(data_dir, "")
#     Save_dir = os.path.split(save_file)[0]
#     if not os.path.exists(Save_dir):
#         os.mkdir(Save_dir)
#     sio.savemat(save_file, {'mask': mask})
#     print test_file, mask.shape
# print "finish"
# print len(test_list)
