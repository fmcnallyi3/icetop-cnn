import tensorflow as tf

# MINI0 - DESIGNED TO TRAIN QUICKLY WITH REASONABLE PERFORMANCE
def get_architecture(inputs, prep):
    ## Create two batch-normalized convolutional layers of kernel size 3 with relu activation
    conv1 = tf.keras.layers.BatchNormalization()(
        tf.keras.layers.Conv2D(16, kernel_size=3, padding='same', activation='relu', data_format='channels_last')(inputs['icetop']))
    conv2 = tf.keras.layers.BatchNormalization()(
        tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', data_format='channels_last')(conv1))

    ## Prepare flattened input to dense layers
    dense_input = tf.keras.layers.Flatten()(conv2)
    if prep['infill']:
        dense_input = tf.keras.layers.Concatenate()([dense_input, tf.keras.layers.Flatten()(inputs['infill'])])
    if prep['reco']:
        dense_input = tf.keras.layers.Concatenate()([dense_input, tf.keras.layers.Flatten()(inputs[prep['reco']])])

    ## Create two batch-normalized dense layers with relu activation
    dense1 = tf.keras.layers.BatchNormalization()(
        tf.keras.layers.Dense(32, activation='relu')(dense_input))
    dense2 = tf.keras.layers.BatchNormalization()(
        tf.keras.layers.Dense(16, activation='relu')(dense1))

    ## Create output tensors
    composition_output = tf.keras.layers.Dense(4, activation='softmax', name='comp')(dense2)
    energy_output = tf.keras.layers.Dense(1, activation='relu', name='energy')(dense2)
    zenith_output = tf.keras.layers.Dense(1, activation='relu', name='zenith')(dense2)
    azimuth_output = tf.keras.layers.Dense(1, activation='relu', name='azimuth')(dense2)
    x_output = tf.keras.layers.Dense(1, activation='relu', name='x_dir')(dense2)
    y_output = tf.keras.layers.Dense(1, activation='relu', name='y_dir')(dense2)
    z_output = tf.keras.layers.Dense(1, activation='relu', name='z_dir')(dense2)
    
    return [composition_output, energy_output, zenith_output, azimuth_output, x_output, y_output, z_output]