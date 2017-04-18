import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import math
import numpy as np
import messlkeras as mk
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Lambda, Merge
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed, Bidirectional
import re
from keras.callbacks import EarlyStopping
import scipy.io as sio
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))
#export CUDA_VISIBLE_DEVICES="1"
import keras

print "Tensorflow version ", tf.__version__
print "Keras version ", keras.__version__

model_dir = '/scratch/felix/IS2017-CHIME3-Exps/exp2-Hyperas-Mask-Cleaner/combo2mask/'
data_dir = '/home/data/CHiME3/data/audio/16kHz/local/messl-spects-noisy/data/'
messl_dir = "/scratch/near/CHiME3/messl_output/"
save_dir = '/scratch/near/CHiME3/v4/lstm_output_max_0406:12:03/'
spect_list = mk.prep_list_for_keras(data_dir, '[ed]t05_.*_((simu)|(real)).*')
mask_list = mk.prep_list_for_keras(messl_dir,'[ed]t05_.*_((simu)|(real)).*')
if [x.split('/')[-1] for x in spect_list]\
	== [x.split('/')[-1] for x in mask_list]:
    print "Training Filenames match! Number of files: {}".format(len(mask_list))
else:
    raise Exception("Training Filenames do not match! Exiting")

weights_filename = 'weights_19:14:18_e:05_vl:0.523657.hdf5'
feat_num = 513
print "loading model...."
# spect sequential model
#model_nspect = Sequential()
# conversion to dB  # f = lambda x: tf.log(tf.abs(x))
#model_nspect.add(TimeDistributed(Lambda(lambda x: tf.log(tf.abs(x))), input_shape = (None,feat_num))) 
# normalize per feature per batch
#model_nspect.add(BatchNormalization(mode=0, axis=2, input_shape = (None,513)))

# MESSL mask seq model
#model_mmask = Sequential()
# normalize per feature per batch
#model_mmask.add(BatchNormalization(mode=0, axis=2, input_shape = (None,513)))

# merge mmask with spect
#model_combo2mask = Sequential()
#model_combo2mask.add(Merge([model_nspect,model_mmask], mode='concat'))

# bidirectional lstm layer
#model_combo2mask.add(Bidirectional(LSTM(input_length=None, output_dim=1024, return_sequences=True)))
#model_combo2mask.add(Bidirectional(LSTM(input_length=None, output_dim=1024, return_sequences=True)))
# out layer
#model_combo2mask.add(TimeDistributed(Dense(output_dim=513, activation='sigmoid')))
# for a mean squared error regression problem
#model_combo2mask.compile(optimizer='RMSprop',loss='mse')
model_combo2mask= load_model(model_dir+weights_filename)
print "model loaded"
print model_combo2mask.summary()
print "......................"
print "start predicting...."
for i in range(len(spect_list)):
    noisy = sio.loadmat(spect_list[i])['data'][0][0][0]
    mask = sio.loadmat(mask_list[i])['data'][0][0][0]
    noisy = noisy.swapaxes(0,2)
    mask = mask.swapaxes(0,2)
    noisy = abs(noisy)
    output = []
    for j in range(6):
        t1 = noisy[j,:].reshape(1,-1,513)
        t2 = mask[0,:].reshape(1,-1,513)
        test_input = [t1,t2]
        test_output = model_combo2mask.predict(test_input)
        output.append(test_output)
    #combine the six masks to one single mask by taking the average.
    mask = np.max(np.asarray(output),axis=0)
    mask = mask[0,:,:]
    print mask.shape
    #save the mask to mat file
    save_file = save_dir+spect_list[i].replace(data_dir, "")
    Save_dir = os.path.split(save_file)[0]
    if not os.path.exists(Save_dir):
        os.mkdir(Save_dir)
    sio.savemat(save_file, {'mask': mask})
print "finish"

