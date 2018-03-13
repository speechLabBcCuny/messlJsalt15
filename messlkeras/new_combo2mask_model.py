
# imports
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Lambda, concatenate, multiply, Dropout, Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed , Bidirectional
from keras import regularizers
from keras import backend as K



def new_combo2mask_model(   input_length, 
                            exp_type, 
                            layer_sizes=[128,128],
                            bid_merge_mode='ave',
                            out_activation='relu',
                            drop_rate=0.5):

    # returns an uncompiled keras 2.0 model built using the given config (also returns the config)
    # input: a list of configurations parameters
    
    # chime3 specific parameters
    feat_num = 513
    output_dim = 513
    
    config = {'input_length':input_length,
                'layer_sizes':layer_sizes,
                'bid_merge_mode':bid_merge_mode,
                'out_activation':out_activation,
                'drop_rate': drop_rate}

    # many more to be added
    # LSTM parameters
    # regularizers
    
    # input 1: a noisy spectrogram of fixed length
    nspect_inputs = Input(shape=(input_length, feat_num))

    # input 2: MESSL mask
    mmask_inputs = Input(shape=(input_length, feat_num))
    
    # merge the two inputs
    merged_inputs = concatenate([nspect_inputs, mmask_inputs])

    # LSTM layer(s)
    x = Bidirectional(LSTM(units=layer_sizes[0], return_sequences=True), merge_mode=bid_merge_mode)(merged_inputs)
    for size in layer_sizes[1:] :
        x = Bidirectional(LSTM(units=size, return_sequences=True), merge_mode=bid_merge_mode)(x)

    # drop out 
    x = Dropout(rate=drop_rate)(x)

    # output: ie the predicted mask
    mask_preds = Dense(513, activation=out_activation, use_bias=True, kernel_regularizer=regularizers.l2(0.01),  name='mask_preds')(x)
#     mask_preds = Dense(513, activation=out_activation, use_bias = True, name='mask_preds')(x)

    if exp_type in ['psa', 'msa']:
        predictions = multiply([mask_preds, nspect_inputs])
    elif exp_type in ['iaf', 'psf']:
        predictions = mask_preds

    # the final model
    model = Model(inputs=[nspect_inputs,mmask_inputs], outputs=predictions)

    return [config, model]