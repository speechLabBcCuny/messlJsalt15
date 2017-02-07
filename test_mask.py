import messlkeras as mk
import scipy.io as sio
import numpy as np
import keras.backend as K
from keras.models import load_model

base_dir = '/scratch/near/test/mask/'
model_dir = '/home/data/CHiME3/experiments/exp1/exp_2017-02-03_00:38:48/'
save_dir = '/scratch/near/test_prediction/'
data_dir = '/home/data/CHiME3/data/audio/16kHz/local/messl-spects-noisy/data/'
test_list = mk.prep_list_for_keras(data_dir, 'dt05_*_simu')
print len(test_list)
#load the latest model

print "loading model...."
model = load_model(model_dir+'keras_model_nspect2mask.hdf5')
print "model loaded"
print "......................"
print "start predicting...."
for test_file in test_list:
    noisy = sio.loadmat(test_file)['data'][0][0][0]
    noisy = noisy.swapaxis(0,2)
    test_input = abs(noisy)
    test_output = model.predict(test_input)
    #combine the six masks to one single mask by taking the average.
    mask = np.mean(test_output,axis=0)
    #save the mask to mat file
    save_dir = test_file.replace("messl-spects-noisy", "lstm-predicted-mask")
    sio.savemat(, {'mask': mask})
print "finish"