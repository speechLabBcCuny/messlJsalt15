import math
import numpy as np
import messlkeras as mk
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

model_dir = '/home/felix/myscratch/IS2017-CHIME3-Exps/Saved-Experiments/exp_2017-03-01_15:17:44/'
data_dir = '/home/data/CHiME3/data/audio/16kHz/local/messl-spects-noisy/data/'
messl_dir = "/scratch/mim/chime3/messlMcMvdrMrf.2Hard5Lbp4Slate/data/"
save_dir = '/scratch/near/CHiME3/lstm_output_c3/'
spect_list = mk.prep_list_for_keras(data_dir, '[ed]t05(?!(bth).)*')
mask_list = mk.prep_list_for_keras(messl_dir,'[ed]t05(?!(bth).)*')
if [x.split('/')[-1] for x in spect_list]\
	== [x.split('/')[-1] for x in mask_list]:
    print "Training Filenames match! Number of files: {}".format(len(noisy_spects_list_tr))
else:
    raise Exception("Training Filenames do not match! Exiting")

weights_filename = 'keras_model_combo2mask_weights.h5'

print "loading model...."
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
model.load_weights(model_dir+weights_filename)
print "model loaded"
print "......................"
print "start predicting...."
for i in range(len(test_list)):
    noisy = sio.loadmat(test_list[i])['data'][0][0][0]
    mask = sio.loadmat(mask_list[i])['data'][0][0][0]
    noisy = noisy.swapaxes(0,2)
    mask = mask.swapaxes(0,2)
    test_input = [abs(noisy),mask]
    test_output = model.predict(test_input)
    #combine the six masks to one single mask by taking the average.
    mask = np.mean(test_output,axis=0)
    #save the mask to mat file
    save_file = save_dir+test_file.replace(data_dir, "")
    Save_dir = os.path.split(save_file)[0]
    if not os.path.exists(Save_dir):
        os.mkdir(Save_dir)
    sio.savemat(save_file, {'mask': mask})
print "finish"
print len(test_list)
