import tensorflow as tf

# ADAMW5 - ARCHITECTURE OF BEST KNOWN MODEL TO DATE
# MARGINAL IMPROVEMENT OVER SMALLER MODELS | RAN WITHOUT INFILL
def get_architecture(inputs, prep):
    
    ## Create convolutional blocks with maxpooling
    conv1 = tf.keras.layers.BatchNormalization()(
        tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu', data_format='channels_last')(inputs['icetop']))
    conv2 = tf.keras.layers.BatchNormalization()(
        tf.keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu', data_format='channels_last')(conv1))
    maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=1, padding='same')(conv2)

    conv3 = tf.keras.layers.BatchNormalization()(
        tf.keras.layers.Conv2D(256, kernel_size=3, padding='same', activation='relu', data_format='channels_last')(maxpool1))
    conv4 = tf.keras.layers.BatchNormalization()(
        tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', activation='relu', data_format='channels_last')(conv3))
    maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='same')(conv4)

    conv5 = tf.keras.layers.BatchNormalization()(
        tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', activation='relu', data_format='channels_last')(maxpool2))
    maxpool3 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=1, padding='same')(conv5)

    ## Prepare flattened input to dense layers
    dense_input = tf.keras.layers.Flatten()(maxpool3)
    if prep['infill']:
        dense_input = tf.keras.layers.Concatenate()([dense_input, tf.keras.layers.Flatten()(inputs['infill'])])
    if prep['reco']:
        dense_input = tf.keras.layers.Concatenate()([dense_input, tf.keras.layers.Flatten()(inputs[prep['reco']])])

    ## Create dense layers with dropout
    dense1 = tf.keras.layers.BatchNormalization()(
        tf.keras.layers.Dense(512, activation='relu', kernel_constraint=tf.keras.constraints.MaxNorm(3))(dense_input))
    dropout1 = tf.keras.layers.Dropout(0.2)(dense1)
    dense2 = tf.keras.layers.BatchNormalization()(
        tf.keras.layers.Dense(512, activation='relu', kernel_constraint=tf.keras.constraints.MaxNorm(3))(dropout1))
    dropout2 = tf.keras.layers.Dropout(0.2)(dense2)

    ## Create output tensors
    composition_output = tf.keras.layers.Dense(4, activation='softmax', name='comp')(dropout2)
    energy_output = tf.keras.layers.Dense(1, activation='relu', name='energy')(dropout2)
    zenith_output = tf.keras.layers.Dense(1, activation='relu', name='zenith')(dropout2)
    azimuth_output = tf.keras.layers.Dense(1, activation='relu', name='azimuth')(dropout2)
    x_output = tf.keras.layers.Dense(1, activation='relu', name='x_dir')(dropout2)
    y_output = tf.keras.layers.Dense(1, activation='relu', name='y_dir')(dropout2)
    z_output = tf.keras.layers.Dense(1, activation='relu', name='z_dir')(dropout2)

    return [composition_output, energy_output, zenith_output, azimuth_output, x_output, y_output, z_output]