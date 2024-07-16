# MODEL FILE
# Should only contain the method 'get_compiled_model()'
# Used to avoid bloating the training script and to easily share model architectures
# Will be changed often as model architectures are experimented with

import tensorflow as tf

def get_compiled_model(input_shapes, model_name, prep, predictions):
    
    # Converts scalars to a shape of size 1
    input_shapes = {input_name: shape if shape else (1,) for input_name, shape in input_shapes.items()}

    # Create dictionary of input tensors
    inputs = {input_name: tf.keras.layers.Input(shape=shape) for input_name, shape in input_shapes.items()}

    # MINI0 - DESIGNED TO TRAIN QUICKLY WITH REASONABLE PERFORMANCE
    ## Create two batch-normalized convolutional layers of kernel size 3 with relu activation
    conv1 = tf.keras.layers.BatchNormalization()(
        tf.keras.layers.Conv2D(16, kernel_size=3, padding='same', activation='relu', data_format='channels_last')(inputs['icetop'])
    )
    conv2 = tf.keras.layers.BatchNormalization()(
        tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', data_format='channels_last')(conv1)
    )

    ## Prepare flattened input to dense layers
    dense_input = tf.keras.layers.Flatten()(conv2)
    if prep['infill']:
        dense_input = tf.keras.layers.Concatenate()([dense_input, tf.keras.layers.Flatten()(inputs['infill'])])
    if prep['reco']:
        dense_input = tf.keras.layers.Concatenate()([dense_input, tf.keras.layers.Flatten()(inputs[prep['reco']])])

    ## Create two batch-normalized dense layers with relu activation
    dense1 = tf.keras.layers.BatchNormalization()(
        tf.keras.layers.Dense(64, activation='relu')(dense_input)
    )
    dense2 = tf.keras.layers.BatchNormalization()(
        tf.keras.layers.Dense(32, activation='relu')(dense1)
    )

    ## Create output tensors
    composition_output = tf.keras.layers.Dense(4, activation='softmax', name='comp')(dense2)
    energy_output = tf.keras.layers.Dense(1, activation='relu', name='energy')(dense2)

    outputs = [composition_output, energy_output]

    '''
    # ADAMW5 - ARCHITECTURE OF BEST KNOWN MODEL TO DATE - MARGINAL IMPROVEMENT OVER SMALLER MODELS | RAN WITHOUT INFILL
    conv1 = BatchNormalization()(
        Conv2D(64, kernel_size=3, padding='same', activation='relu', data_format='channels_last')(inputs['icetop']))
    conv2 = BatchNormalization()(
        Conv2D(128, kernel_size=3, padding='same', activation='relu', data_format='channels_last')(conv1))
    maxpool1 = MaxPooling2D(pool_size=2, strides=1, padding='same')(conv2)

    conv3 = BatchNormalization()(
        Conv2D(256, kernel_size=3, padding='same', activation='relu', data_format='channels_last')(maxpool1))
    conv4 = BatchNormalization()(
        Conv2D(512, kernel_size=3, padding='same', activation='relu', data_format='channels_last')(conv3))
    maxpool2 = MaxPooling2D(pool_size=2, strides=2, padding='same')(conv4)

    conv5 = BatchNormalization()(
        Conv2D(512, kernel_size=3, padding='same', activation='relu', data_format='channels_last')(maxpool2))
    maxpool3 = MaxPooling2D(pool_size=2, strides=1, padding='same')(conv5)

    dense_input = Flatten()(maxpool3)
    if prep['infill']:
        dense_input = Concatenate()([dense_input, Flatten()(inputs['infill'])])
    if prep['reco']:
        dense_input = Concatenate()([dense_input, Flatten()(inputs[prep['reco']])])

    dense1 = BatchNormalization()(
        Dense(512, activation='relu', kernel_constraint=MaxNorm(3))(dense_input))
    dropout1 = Dropout(0.2)(dense1)
    dense2 = BatchNormalization()(
        Dense(512, activation='relu', kernel_constraint=MaxNorm(3))(dropout1))
    dropout2 = Dropout(0.2)(dense2)

    composition_output = tf.keras.layers.Dense(4, activation='softmax', name='comp')(dropout2)
    energy_output = tf.keras.layers.Dense(1, activation='relu', name='energy')(dropout2)
    outputs = [composition_output, energy_output]
    '''

    # Create model
    model = tf.keras.models.Model(
        inputs=inputs,
        outputs=[output for output in outputs if output.name[:output.name.index('/')] in predictions],
        name=model_name
    )
    loss_functions = {
        'comp': tf.keras.losses.CategoricalCrossentropy(),
        'energy': tf.keras.losses.Huber()
    }
    metrics = {
        'comp': [tf.keras.metrics.CategoricalAccuracy()],
        'energy': [tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()]
    }
    model.compile( # Compile model
        optimizer=tf.keras.optimizers.Adam(),
        loss={prediction: loss_functions[prediction] for prediction in predictions},
        metrics={prediction: metrics[prediction] for prediction in predictions}
    )
    model.summary()
    return model