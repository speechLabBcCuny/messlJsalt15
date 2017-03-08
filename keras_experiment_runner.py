
# coding: utf-8

# In[1]:

### Keras Experiment Runner
# Train model to produce masks
# 1 - uses hyper-parameter exploration (see below for list)
# 2 - uses early stopping to explore parameter space quickly (stop when validation loss no longer improves)

# potential hyper-parameter space
        # size of LSTM output [256,512,1024,2048])
        # make LSTM bidirectional or not
        # number of LSTM layers [1,2,3,more?]
        # merge_mode for bidirectional LSTM ['sum', 'mul', 'concat', 'ave', None]
        # activation function of output Dense layer: [softmax, softplus, softsign, relu, tanh, sigmoid, hard_sigmoid, linear]
        # loss for whole model: ['mean_squared_error / mse', 'mean_absolute_error / mae', 'mean_absolute_percentage_error / mape', 'mean_squared_logarithmic_error / msle'
        # (continued) squared_hinge, hinge, binary_crossentropy, kullback_leibler_divergence, poisson, cosine_proximity]
        # optimizer [SGD, RMSprop, AdaGrad, AdaDelta, Adam, Adam, Adamax, Nadam]
        # 'batch_size [32, 64, 128, 256, 512]


# In[2]:

from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, conditional

# from keras.models import Sequential
# from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.layers import LSTM, Dense, Lambda, Merge
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))

import time
import simplejson


# In[3]:

# print experiment time for logging
print("Running experiment.")
start_time = time.strftime('%Y-%m-%d %T')
print(start_time)


# In[4]:

# # create new experiment folder
# save_dir = "/home/felix/Desktop/PhD Work/CHIME Results/temp/"
# newexp_folder_path = save_dir+'/'+"exp_"+start_time+'/'
# print("Creating new folder for this experiment:", newexp_folder_path)

# os.makedirs(newexp_folder_path)


# In[5]:

def data():
   
    # add messlKeras (why do I need to reimport these?)
    import messlkeras as mk
    from keras import backend as K

    #folder with the data
    data_dir = "/home/data/CHiME3/data/audio/16kHz/local/"
    
    # training data
    noisy_spects_list_tr = mk.prep_list_for_keras(data_dir, "messl-spects-noisy.*tr05.*real", verbose=False)
    masks_list_tr = mk.prep_list_for_keras(data_dir, "mask.*ideal_amplitude.*tr05.*real", verbose=False)

    # check that they match:
    if [x.split('/')[-1] for x in noisy_spects_list_tr] == [x.split('/')[-1] for x in masks_list_tr]:
        pass
    else:
        raise Exception("Filenames do not match! Exiting")
    
    # validation data
    noisy_spects_list_et = mk.prep_list_for_keras(data_dir, "messl-spects-noisy.*dt05.*real", verbose=False)
    masks_list_et = mk.prep_list_for_keras(data_dir, "mask.*ideal_amplitude.*dt05.*real", verbose=False)

    # check that they match:
    if [x.split('/')[-1] for x in noisy_spects_list_et] == [x.split('/')[-1] for x in masks_list_et]:
        pass
    else:
        raise Exception("Filenames do not match! Exiting")
    
    ### prepare data
    sample_num, input_length, feat_num = (-1,150,513)
    input_shape = (sample_num, input_length, feat_num)
    start = 0

    # prep training data
    keras_m1_inputs_tr, num_proc_files1  = mk.prep_data_for_keras(noisy_spects_list_tr, input_shape, start, time_limit=7200)
    keras_m1_targets_tr, num_proc_files2 = mk.prep_data_for_keras(masks_list_tr, input_shape, start, time_limit=7200)

    # prep early stopping data
    keras_m1_inputs_et, num_proc_files1  = mk.prep_data_for_keras(noisy_spects_list_et, input_shape, start, time_limit=7200)
    keras_m1_targets_et, num_proc_files2 = mk.prep_data_for_keras(masks_list_et, input_shape, start, time_limit=7200)

    X_train = abs(keras_m1_inputs_tr)
    Y_train = keras_m1_targets_tr
    X_test = abs(keras_m1_inputs_et)
    Y_test = keras_m1_targets_et
        
    return X_train, Y_train, X_test, Y_test

# In[7]:

def model(X_train, Y_train, X_test, Y_test):

    model_nspect2imask = Sequential()
    # parameters of the lstm
    feat_num = 513
    output_dim = 513

    # conversion to dB  # f = lambda x: tf.log(tf.abs(x))
    model_nspect2imask.add(TimeDistributed(Lambda(lambda x: K.log(K.abs(x))), input_shape = (None,feat_num))) 

    # normalize per feature per batch
    model_nspect2imask.add(BatchNormalization(mode=0, axis=-1, input_shape = (None,513)))

    # bidirectional lstm layer
    # layer size to be optimized
    model_nspect2imask.add(Bidirectional(LSTM(input_length=None, output_dim={{choice([256,512,1024,2048])}}, return_sequences=True), merge_mode={{choice(['sum', 'mul', 'concat', 'ave'])}} ))
    # potentially add a layer
    if conditional({{choice(['two layers', 'three layers'])}}) == 'three layers':
        model_nspect2imask.add(Bidirectional(LSTM(input_length=None, output_dim={{choice([256,512,1024,2048])}}, return_sequences=True), merge_mode={{choice(['sum', 'mul', 'concat', 'ave'])}} ))


    # out layer
    # activation function to be optimized
    model_nspect2imask.add(TimeDistributed(Dense(output_dim=output_dim, activation={{choice(['sigmoid', 'hard_sigmoid'])}} )))
    # for a mean squared error regression problem
    model_nspect2imask.compile(optimizer={{choice(['SGD', 'RMSprop', 'AdaGrad', 'AdaDelta', 'Adam', 'Adamax', 'Nadam'])}},loss={{choice(['mse', 'binary_crossentropy']) }})
    
    # callbacks
    early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=0, mode='auto')
    
    # hardcode the save path for lack of better solution
    cur_time = time.strftime('%T')
    exp_folder_path = "/home/felix/myscratch/IS2017-CHIME3-Exps/exp-Hyperas-LSTM-nspect2imask/2017-03-07_14:10"
    filepath = exp_folder_path + "/" + "weights_"+cur_time+"_e:{epoch:02d}_vl:{val_loss:.4f}.hdf5"
    model_checker = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=0)
    
    callbacks = [model_checker, early_stopper]

    model_nspect2imask.fit(X_train, Y_train, batch_size={{choice([32, 64, 128, 256, 512])}}, nb_epoch=100, shuffle=True, verbose=2, validation_data=(X_test, Y_test),                          callbacks = callbacks)
    
#     score, val_loss = model_nspect2imask.evaluate(X_test, Y_test, verbose=0)
#     print('val_loss:', val_loss)
    print("Run done.")
    end_time = time.strftime('%Y-%m-%d %T')
    print(end_time)
    return {'loss': 1, 'status': STATUS_OK, 'model': model_nspect2imask}


# In[ ]:

# model(X_train, Y_train, X_test, Y_test)


# In[ ]:

X_train, Y_train, X_test, Y_test = data()

best_run, best_model = optim.minimize(model=model, data=data, algo=tpe.suggest, max_evals=1000, trials=Trials())


# In[ ]:

print("Evaluation of best performing model:")
print(best_model.evaluate(X_test, Y_test))

print("Saving the best model")
# save model to file

# save model to file
exp_folder_path = "/home/felix/myscratch/IS2017-CHIME3-Exps/exp-Hyperas-LSTM-nspect2imask/2017-03-07_14:10/"
model_filename = "keras_bestmodel_nspect2imask.json"

# serialize model to JSON
best_model_json = best_model.to_json()
with open(exp_folder_path+'/'+model_filename, "w") as json_file:
    json_file.write(simplejson.dumps(simplejson.loads(best_model_json), indent=4))

# serialize weights to HDF5
weights_filename = "keras_bestmodel_nspect2imask_weights.h5"
best_model.save_weights(exp_folder_path+'/'+weights_filename)
print("Saved model json and weights to disk")
