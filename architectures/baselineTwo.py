import tensorflow as tf

CONV1_FILTERS = 64    # ↑ captures more low-level features, ↑ compute cost; ↓ less detail, faster
CONV2_FILTERS = 128   # ↑ learns richer patterns, ↑ memory; ↓ simpler, may underfit
CONV3_FILTERS = 256
CONV4_FILTERS = 512
KERNEL_SIZE = 3       # ↑ sees broader context; ↓ focuses on fine details
PADDING = 'same'      # 'same' keeps spatial size; 'valid' shrinks output
ACTIVATION='relu'     # Controls non-linearity in conv layers; affects feature scaling

DENSE1_UNITS = 128    # ↑ more capacity to learn from features; ↓ faster, may underfit
DENSE2_UNITS = 64     # Same as above, deeper in network
DENSE3_UNITS = 32
MAXNORM = 3           # ↓ stronger regularization; ↑ weaker constraint, more flexibility

DROPOUT_RATE = 0.35   # ↑ reduces overfitting but slows learning; ↓ faster, may overfit

#DIFFERENCES FROM BASELINE ONE:
#More clear how to change the variables
#Added two extra convultional layers
#Increased the overall conv. layer filters
#Added a third dense layer
#Added in more clear (lower?) dropout
#How does this change things? IDK! 

# BASELINETWO - 
def get_architecture(inputs, prep):

    ## Create convolutional layers
    conv1 = tf.keras.layers.BatchNormalization()(
        tf.keras.layers.Conv2D(CONV1_FILTERS, kernel_size=KERNEL_SIZE, padding=PADDING, activation=ACTIVATION, data_format='channels_last')(inputs['icetop']))
    conv2 = tf.keras.layers.BatchNormalization()(
        tf.keras.layers.Conv2D(CONV2_FILTERS, kernel_size=KERNEL_SIZE, padding=PADDING, activation=ACTIVATION, data_format='channels_last')(conv1))
    maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=1, padding='same')(conv2)

    conv3 = tf.keras.layers.BatchNormalization()(
        tf.keras.layers.Conv2D(CONV3_FILTERS, kernel_size=KERNEL_SIZE, padding=PADDING, activation=ACTIVATION, data_format='channels_last')(maxpool1))
    conv4 = tf.keras.layers.BatchNormalization()(
        tf.keras.layers.Conv2D(CONV4_FILTERS, kernel_size=KERNEL_SIZE, padding=PADDING, activation=ACTIVATION, data_format='channels_last')(conv3))
    maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='same')(conv4)

    ## Prepare flattened input to dense layers
    dense_input = tf.keras.layers.Flatten()(maxpool2)
    if prep['infill']:
        dense_input = tf.keras.layers.Concatenate()([dense_input, tf.keras.layers.Flatten()(inputs['infill'])])
    if prep['reco']:
        dense_input = tf.keras.layers.Concatenate()([dense_input, tf.keras.layers.Flatten()(inputs[prep['reco']])])

    ## Create dense layers with dropout
    dense1 = tf.keras.layers.BatchNormalization()(
        tf.keras.layers.Dense(DENSE1_UNITS, activation=ACTIVATION, kernel_constraint=tf.keras.constraints.MaxNorm(MAXNORM))(dense_input))
    dropout1 = tf.keras.layers.Dropout(DROPOUT_RATE)(dense1)
    dense2 = tf.keras.layers.BatchNormalization()(
        tf.keras.layers.Dense(DENSE2_UNITS, activation=ACTIVATION, kernel_constraint=tf.keras.constraints.MaxNorm(MAXNORM))(dropout1))
    dropout2 = tf.keras.layers.Dropout(DROPOUT_RATE)(dense2)
    dense3 = tf.keras.layers.BatchNormalization()(
        tf.keras.layers.Dense(DENSE3_UNITS, activation=ACTIVATION, kernel_constraint=tf.keras.constraints.MaxNorm(MAXNORM))(dropout2))
    dropout3 = tf.keras.layers.Dropout(DROPOUT_RATE)(dense3)
    

    ## Create output tensors
    composition_output = tf.keras.layers.Dense(4, activation='softmax', name='comp')(dropout3)
    energy_output = tf.keras.layers.Dense(1, activation='relu', name='energy')(dropout3)

    return [composition_output, energy_output]