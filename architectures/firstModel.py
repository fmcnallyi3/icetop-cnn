import tensorflow as tf


#Model Parameters
# Convolutional layers
CONV1_FILTERS = 256           # ↑ captures more low-level features, ↑ compute cost; ↓ less detail, faster
CONV2_FILTERS = 512           # ↑ learns richer patterns, ↑ memory; ↓ simpler, may underfit
KERNEL_SIZE = 3               # ↑ sees broader context; ↓ focuses on fine details
PADDING_TYPE = 'same'         # 'same' keeps spatial size; 'valid' shrinks output
CONV_ACTIVATION = 'relu'      # Controls non-linearity in conv layers; affects feature scaling

DENSE_UNITS_1 = 128           # ↑ more capacity to learn from features; ↓ faster, may underfit
DENSE_UNITS_2 = 128           # Same as above, deeper in network
DENSE_UNITS_3 = 128           # Same as above, final dense before outputs
DENSE_ACTIVATION = 'relu'     # Controls non-linearity in dense layers; affects learning dynamics
MAX_NORM_CONSTRAINT = 4       # ↓ stronger regularization; ↑ weaker constraint, more flexibility

DROPOUT_RATE = 0.65           # ↑ reduces overfitting but slows learning; ↓ faster, may overfit

NUM_CLASSES = 4               # Must match dataset class count; changes classification output size
CLASS_ACTIVATION = 'softmax'  # 'softmax' for multi-class; 'sigmoid' for multi-label
REG_ACTIVATION = 'relu'       # 'relu' forces non-negative regression; 'linear' allows negatives


#Model Architecture 
def get_architecture(inputs, prep):
    # Convolutional feature extractor
    conv1 = tf.keras.layers.BatchNormalization()(inputs)
    conv1 = tf.keras.layers.Conv2D(
        CONV1_FILTERS,
        kernel_size=KERNEL_SIZE,
        padding=PADDING_TYPE,
        activation=CONV_ACTIVATION,
        data_format='channels_last'
    )(conv1)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv1 = tf.keras.layers.Conv2D(
        CONV2_FILTERS,
        kernel_size=KERNEL_SIZE,
        padding=PADDING_TYPE,
        activation=CONV_ACTIVATION,
        data_format='channels_last'
    )(conv1)

    # Flatten and concatenate CNN features with prep input
    dense_input = tf.keras.layers.Concatenate()([
        tf.keras.layers.Flatten()(conv1),
        tf.keras.layers.Flatten()(prep)
    ])
    dense_input = tf.keras.layers.BatchNormalization()(dense_input)

    # Dense layers
    reco = tf.keras.layers.Dense(
        DENSE_UNITS_1,
        activation=DENSE_ACTIVATION,
        name='reco1'
    )(dense_input)
    reco = tf.keras.layers.Dense(
        DENSE_UNITS_2,
        activation=DENSE_ACTIVATION,
        kernel_constraint=tf.keras.constraints.MaxNorm(MAX_NORM_CONSTRAINT)
    )(reco)

    dense2 = tf.keras.layers.BatchNormalization()(reco)
    dense2 = tf.keras.layers.Dense(
        DENSE_UNITS_3,
        activation=DENSE_ACTIVATION,
        kernel_constraint=tf.keras.constraints.MaxNorm(MAX_NORM_CONSTRAINT)
    )(dense2)
    dense2 = tf.keras.layers.Dropout(DROPOUT_RATE)(dense2)

    # Output layers
    composition_output = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation=CLASS_ACTIVATION,
        name='comp'
    )(dense2)
    energy_output = tf.keras.layers.Dense(
        1,
        activation=REG_ACTIVATION,
        name='energy'
    )(dense2)

    return composition_output, energy_output