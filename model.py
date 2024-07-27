# MODEL FILE
# Should only contain the method 'get_compiled_model()'

import importlib.util
import os

import tensorflow as tf

def get_compiled_model(input_shapes, model_name, model_design, prep, predictions):
    
    # Converts scalars to a shape of size 1
    input_shapes = {input_name: shape if shape else (1,) for input_name, shape in input_shapes.items()}

    # Create dictionary of input tensors
    inputs = {input_name: tf.keras.layers.Input(shape=shape) for input_name, shape in input_shapes.items()}

    # Import desired model architecture
    spec = importlib.util.spec_from_file_location(model_design, f'{os.getenv("ICETOP_CNN_DIR")}/architectures/{model_design}.py')
    arch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arch)

    # Load desired model architecture
    outputs = arch.get_architecture(inputs, prep)

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
        'energy': [None]
    }
    model.compile( # Compile model
        optimizer=tf.keras.optimizers.Adam(),
        loss={prediction: loss_functions[prediction] for prediction in predictions},
        metrics={prediction: metrics[prediction] for prediction in predictions}
    )
    model.summary()
    return model