import tensorflow as tf

CONV1_FILTERS = 64    # ↑ captures more low-level features, ↑ compute cost; ↓ less detail, faster
CONV2_FILTERS = 128   # ↑ learns richer patterns, ↑ memory; ↓ simpler, may underfit

KERNEL_SIZE = 3       # ↑ sees broader context; ↓ focuses on fine details
PADDING = 'same'      # 'same' keeps spatial size; 'valid' shrinks output
ACTIVATION='relu'     # Controls non-linearity in conv layers; affects feature scaling

DENSE1_UNITS = 128    # ↑ more capacity to learn from features; ↓ faster, may underfit
DENSE2_UNITS = 64     # Same as above, deeper in networ

MAXNORM = 3           # ↓ stronger regularization; ↑ weaker constraint, more flexibility

DROPOUT_RATE = 0.25   # ↑ reduces overfitting but slows learning; ↓ faster, may overfit

#DIFFERENCES FROM BASELINE ONE:
#The same amount of layers as BL1, 
#It just has increased filter and unit counts
#Also lower dropout

# js3 - 
def get_architecture(inputs, prep):

    ## Create convolutional layers
    conv1 = tf.keras.layers.BatchNormalization()(
        tf.keras.layers.Conv2D(CONV1_FILTERS, kernel_size=KERNEL_SIZE, padding=PADDING, activation=ACTIVATION, data_format='channels_last')(inputs['icetop']))
    conv2 = tf.keras.layers.BatchNormalization()(
        tf.keras.layers.Conv2D(CONV2_FILTERS, kernel_size=KERNEL_SIZE, padding=PADDING, activation=ACTIVATION, data_format='channels_last')(conv1))

    ## Prepare flattened input to dense layers
    dense_input = tf.keras.layers.Flatten()(conv2)
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
    
    ## Create output tensors
    composition_output = tf.keras.layers.Dense(4, activation='softmax', name='comp')(dropout2)
    energy_output = tf.keras.layers.Dense(1, activation='relu', name='energy')(dropout2)

    return [composition_output, energy_output]