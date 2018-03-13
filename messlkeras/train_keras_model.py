from keras.callbacks import EarlyStopping, TerminateOnNaN, ModelCheckpoint
from keras.optimizers import Nadam

def train_keras_model(model, save_dir,
                        tr_inputs, tr_targets, val_inputs, val_targets,
                        optimizer='nadam', loss='binary_crossentropy', batch_size=64, epochs=10,
                        patience=10, period=50):
    ### trains a single keras model
    # inputs: uncompiled model,training and validation data, experiment configuration
    # side effects, backup model weights are saved in  save_dir
    # output: history, model

    # fitting parameters
    # optimizer parameters
    lr=0.001
    if optimizer=='nadam':
        optimizer = Nadam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    # TODO add the rest
    

    model.compile(optimizer=optimizer, loss=loss)

    # callbacks
    early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=0, mode='auto')
    terminate_on_nan = TerminateOnNaN()
    # model_checker = ModelCheckpoint(save_dir+'weights.ep{epoch:02d}-vl{val_loss:.4f}.hdf5', 
    model_checker = ModelCheckpoint(save_dir+'/best_model_weights.hdf5', 
        monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=period)


    callbacks = [terminate_on_nan, early_stopper, model_checker]

    # fitting (training)
    hist = model.fit(tr_inputs, tr_targets,
                     batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2, 
                     validation_data=(val_inputs, val_targets), callbacks=callbacks)

    return hist, model
