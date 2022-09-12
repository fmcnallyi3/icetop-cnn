from csv import writer
from data_tools import data_prep, get_data_cut, load_preprocessed
from keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau
from keras.layers import BatchNormalization, Concatenate, Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import tensorflow as tf

# Set GPU to train on
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'
# Set dynamic memory allocation
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def main():
    ### Baseline data prep ###

    # Edit these parameters
    # Too many to list, look in data_prep in data_tools for a better idea of what each does
    prep = {'clc':True, 'sta5':False, 'q':None, 't':None, 't_shift':True, 't_clip':0, 'normed':True, 'reco':None, 'cosz':False}

    # Set the number of models to train under this CNN
    num_models_to_train = 1

    # Name for model(s)
    model_name = 'baseline'

    # Type of model to train
    model_type = 'bauwens'

    # Set the number of epochs the model(s) should run for
    # Actual result may differ due to early stopping
    num_epochs = 200

    # Loss metric to use for training
    # Suggestion to experiment with 'huber_loss'
    loss_function = 'huber_loss'

    # Optimizer to user for training
    # Default lr is .001
    optimizer = Adam(learning_rate=0.001)

    # Other loss metrics to analyze while training
    # Only for user to monitor - have no effect on model training
    metrics = ['mae','mse']

    # File directory to folder that holds models
    model_prefix = os.getcwd()+'/models'

    # File directory to folder that holds simulation data 
    sim_prefix = os.getcwd()+'/simdata'

    # Booleans for easier to read conditionals - no need to change this
    has_reco, has_time = prep['reco'] != None, prep['t'] != False


    # Load simulation data from files for training
    x, y = load_preprocessed(sim_prefix, comp=['p','h','o','f'])

    # Prepare simulation data
    x_i, idx, pre_cut = data_prep(x, y, 'train', **prep)

    ### Cut data ###
    data_cut = get_data_cut(prep['reco'], y, pre_cut)
    
    if has_reco:
        x_i[0] = x_i[0][data_cut] # q/t
        x_i[1] = x_i[1][data_cut] # z
    else:
        x_i = x_i[data_cut] # q/t


    for num_model in range(num_models_to_train):

        model, name, fit_inputs = create_model(model_name, model_prefix, model_type, has_time, has_reco, x_i)
        model.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)
        # Print summary of model. Can be commented out or removed if the text is too much.
        model.summary()

        # Arguments to play with are factor (best between 0.1 - 0.8), patience, and min_lr
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=10, mode='min', min_lr=0.0001)
        # Only argument to play with is patience. Recommended to be greater than twice the reduce_lr patience.
        early_stop = EarlyStopping(monitor='val_loss', patience=25, mode='min', restore_best_weights=True)
        csv_logger = CSVLogger('%s/%s' % (model_prefix, name))

        history = model.fit(fit_inputs, y=y['energy'][data_cut], epochs=num_epochs, validation_split=0.15, callbacks=[early_stop, csv_logger, reduce_lr])


        model.save('%s/%s.h5' % (model_prefix, name))
        np.save('%s/%s.npy' % (model_prefix, name), prep)

        
        val_loss = np.min(history.history['val_loss'])
        index = history.history['val_loss'].index(val_loss)
        loss = history.history['loss'][index]
        new_row = [name, index, loss, val_loss]
        with open('models/results.csv', 'a') as f:
            writer(f).writerow(new_row)
        f.close() 

def create_model(model_name, model_prefix, model_type, has_time, has_reco, x_i):

    if not has_time and not has_reco:
        raise Exception('Why train the model on charge alone? It is not worth it, promise.')

    # Ensures models are not overwritten
    name = model_name
    i = 0
    while(os.path.exists('%s/%s' % (model_prefix, name+str(i)))): i += 1
    name += str(i)

    # Charge and Time (if included) input layers
    data_input = Input(shape=x_i[0].shape[-3:], name='data')

    # Zenith input layer
    if has_reco:
        zenith_input = Input(shape=(1), name='zenith')


    if model_type == 'baseline':
        conv1 = Conv2D(64, kernel_size=3, padding='same', activation='relu', data_format='channels_last')(data_input)
        conv2 = Conv2D(32, kernel_size=3, padding='same', activation='relu', data_format='channels_last')(conv1)
        conv3 = Conv2D(16, kernel_size=3, padding='same', activation='relu', data_format='channels_last')(conv2)
        flat = Flatten()(conv3)
        # Must concatenate Zenith input to Flat layer
        if has_reco:
            flat = Concatenate()([flat, zenith_input])
        dense1 = Dense(256, activation='relu')(flat)
        dense2 = Dense(256, activation='relu')(dense1)
        dense3 = Dense(256, activation='relu')(dense2)
        output = Dense(1, activation='relu')(dense3)
    
    elif model_type == 'bauwens':   
        conv1 = BatchNormalization()(Conv2D(64, kernel_size=3, padding='same', activation='relu', data_format='channels_last')(data_input))
        conv2 = BatchNormalization()(Conv2D(128, kernel_size=3, padding='same', activation='relu', data_format='channels_last')(conv1))
        maxpool1 = MaxPooling2D(pool_size=3, strides=2, padding='same')(conv2)
        conv3 = BatchNormalization()(Conv2D(256, kernel_size=3, padding='same', activation='relu', data_format='channels_last')(maxpool1))
        conv4 = BatchNormalization()(Conv2D(512, kernel_size=3, padding='same', activation='relu', data_format='channels_last')(conv3))
        maxpool2 = MaxPooling2D(pool_size=3, strides=2, padding='same')(conv4)
        conv5 = BatchNormalization()(Conv2D(512, kernel_size=3, padding='same', activation='relu', data_format='channels_last')(maxpool2))
        maxpool3 = MaxPooling2D(pool_size=3, strides=2, padding='same')(conv5)
        flat = Flatten()(maxpool3)
        # Must concatenate Zenith input to Flat layer
        if has_reco:
            flat = Concatenate()([flat, zenith_input])
        dense1 = BatchNormalization()(Dense(1024, activation='relu')(flat))
        output = Dense(1, activation='relu')(dense1)

    else:
        raise Exception('Unrecognized model type.')


    inputs = [data_input]
    fit_inputs = {'data':x_i}
    if has_reco:
        inputs.append(zenith_input)
        fit_inputs = {'data':x_i[0], 'zenith':x_i[1].reshape(-1,1)}
    model = Model(inputs=inputs, outputs=output, name=name)

    return model, name, fit_inputs

if __name__ == '__main__':
    main()
