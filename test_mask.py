import messlkeras as mk
import scipy.io as sio
import numpy as np
import keras.backend as K
from keras.models import load_model
import os
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Lambda, Merge
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed, Bidirectional
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))
base_dir = '/scratch/near/test/mask/'
model_dir = '/scratch/felix/IS2017-CHIME3-Exps/exp-nspect2mask/iaf/exp_2017-04-07 22:38:12/'
save_dir = '/scratch/near/CHiME3/fixed_model/v2/lstm-out-0411:14:06/'
data_dir = '/home/data/CHiME3/data/audio/16kHz/local/messl-spects-noisy/data/'
test_list = mk.prep_list_for_keras(data_dir, '[ed]t05')
weights_filename = "iaf_model_2017-04-10 15:10:20_vl:0.576155249803.hdf5"

print len(test_list)
#load the latest model

print "loading model...."
# sequential model
model = Sequential()
# parameters of the lstm
# sample_num, input_length, feat_num = sample_num, input_length, feat_num // doesn't do anything just a reminder
output_dim = 513
feat_num = 513
# conversion to dB  # f = lambda x: tf.log(tf.abs(x))
#model.add(TimeDistributed(Lambda(lambda x: tf.log(tf.abs(x))), input_shape = (None,feat_num)))

# normalize per feature per batch
#model.add(BatchNormalization(mode=0, axis=-1, input_shape = (None,513)))

# bidirectional lstm layer
#model.add(Bidirectional(LSTM(input_length=None, output_dim=1024, return_sequences=True)))

# out layer
#model.add(TimeDistributed(Dense(output_dim=output_dim, activation='sigmoid')))
# for a mean squared error regression problem
#model.compile(optimizer='RMSprop',loss='mse')
# load weights into new model
model=load_model(model_dir+weights_filename)
print("Loaded model weights from disk")
print "model loaded"
print "......................"
print "start predicting...."
for test_file in test_list:
    noisy = sio.loadmat(test_file)['data'][0][0][0]
    noisy = noisy.swapaxes(0,2)
    test_input = abs(noisy)
    test_output = model.predict(test_input)
    #combine the six masks to one single mask by taking the average.
    mask = np.max(test_output,axis=0)
    #save the mask to mat file
    save_file = save_dir+test_file.replace(data_dir, "")
    Save_dir = os.path.split(save_file)[0]
    if not os.path.exists(Save_dir):
        os.mkdir(Save_dir)
    sio.savemat(save_file, {'mask': mask})
    print test_file, mask.shape
print "finish"
print len(test_list)
