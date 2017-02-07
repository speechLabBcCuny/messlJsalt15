import messlkeras as mk
import scipy.io as sio
import numpy as np
import keras.backend as K
from keras.models import load_model
import os

base_dir = '/scratch/near/test/mask/'
model_dir = '/home/data/CHiME3/experiments/exp1/exp_2017-02-03_00:38:48/'
save_dir = '/scratch/near/test_prediction/'
data_dir = '/home/data/CHiME3/data/audio/16kHz/local/messl-spects-noisy/data/'
test_list = mk.prep_list_for_keras(data_dir, 'dt05.*simu')
print len(test_list)
#load the latest model

print "loading model...."
model = load_model(model_dir+'keras_model_nspect2mask.hdf5')
print "model loaded"
print "......................"
print "start predicting...."
for test_file in test_list:
    noisy = sio.loadmat(test_file)['data'][0][0][0]
    noisy = noisy.swapaxes(0,2)
    test_input = abs(noisy)
    test_output = model.predict(test_input)
    #combine the six masks to one single mask by taking the average.
    mask = np.mean(test_output,axis=0)
    #save the mask to mat file
    save_file = model_dir+test_file.replace(data_dir, "")
    save_dir = os.path.split(save_file)[0]
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    sio.savemat(save_file, {'mask': mask})
print "finish"
print len(test_list)
