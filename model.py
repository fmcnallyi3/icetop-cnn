# MODEL FILE
# Should only contain the method 'get_compiled_model()'
# Used to avoid bloating the training script and to easily share model architectures
# Will be changed often as model architectures are experimented with

from keras.layers import BatchNormalization, Concatenate, Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras.models import Model
from keras.constraints import MaxNorm
from keras.optimizers import Adam

import config as cg

def get_compiled_model(input_shapes):
    
    # Converts scalars to a shape of size 1
    input_shapes = {input_name: shape if shape else (1,) for input_name, shape in input_shapes.items()}

    # Create dictionary of input tensors
    inputs = {input_name: Input(shape=shape) for input_name, shape in input_shapes.items()}

    # MINI0 - DESIGNED TO TRAIN QUICKLY WITH REASONABLE PERFORMANCE
    ## Create two batch-normalized convolutional layers of kernel size 3 with relu activation
    conv1 = BatchNormalization()(Conv2D(16, kernel_size=3, padding='same', activation='relu', data_format='channels_last')(inputs['icetop']))
    conv2 = BatchNormalization()(Conv2D(32, kernel_size=3, padding='same', activation='relu', data_format='channels_last')(conv1))

    ## Prepare flattened input to dense layers
    dense_input = Flatten()(conv2)
    if cg.PREP['infill']:
        dense_input = Concatenate()([dense_input, Flatten()(inputs['infill'])])
    if cg.PREP['reco']:
        dense_input = Concatenate()([dense_input, Flatten()(inputs[cg.PREP['reco']])])

    ## Create two batch-normalized dense layers with relu activation
    dense1 = BatchNormalization()(Dense(32, activation='relu')(dense_input))
    dense2 = BatchNormalization()(Dense(64, activation='relu')(dense1))

    ## Create single-output tensor
    outputs = Dense(1, activation='relu')(dense2)

    model = Model(inputs=inputs, outputs=outputs, name=cg.MODEL_NAME)             # Create model
    model.compile(optimizer=Adam(), loss=cg.LOSS_FUNCTION, metrics=cg.METRICS)    # Compile model
    #model.summary()
    return model

    '''

    # ARCHITECTURE OF BEST KNOWN MODEL TO DATE - MARGINAL IMPROVEMENT OVER SMALLER MODELS | RAN WITHOUT INFILL

    # ADAMW5
    conv1 = BatchNormalization()(Conv2D(64, kernel_size=3, padding='same', activation='relu', data_format='channels_last')(inputs['icetop']))
    conv2 = BatchNormalization()(Conv2D(128, kernel_size=3, padding='same', activation='relu', data_format='channels_last')(conv1))
    maxpool1 = MaxPooling2D(pool_size=2, strides=1, padding='same')(conv2)

    conv3 = BatchNormalization()(Conv2D(256, kernel_size=3, padding='same', activation='relu', data_format='channels_last')(maxpool1))
    conv4 = BatchNormalization()(Conv2D(512, kernel_size=3, padding='same', activation='relu', data_format='channels_last')(conv3))
    maxpool2 = MaxPooling2D(pool_size=2, strides=2, padding='same')(conv4)

    conv5 = BatchNormalization()(Conv2D(512, kernel_size=3, padding='same', activation='relu', data_format='channels_last')(maxpool2))
    maxpool3 = MaxPooling2D(pool_size=2, strides=1, padding='same')(conv5)

    dense_input = Flatten()(maxpool3)
    if cg.PREP['infill']:
        dense_input = Concatenate()([dense_input, Flatten()(inputs['infill'])])
    if cg.PREP['reco']:
        dense_input = Concatenate()([dense_input, Flatten()(inputs[cg.PREP['reco']])])

    dense1 = BatchNormalization()(Dense(512, activation='relu', kernel_constraint=MaxNorm(3))(dense_input))
    dropout1 = Dropout(0.2)(dense1)
    dense2 = BatchNormalization()(Dense(512, activation='relu', kernel_constraint=MaxNorm(3))(dropout1))
    dropout2 = Dropout(0.2)(dense2)
    outputs = Dense(1, activation='relu')(dropout2)

    '''