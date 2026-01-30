import tensorflow_decision_forests as tfdf
import tensorflow as tf

def get_architecture(inputs, prep):
    # Define the hyperparameter tuner
    tuner = tfdf.tuner.RandomSearch(
        num_trials=100,  # Number of random hyperparameter configurations to evaluate
        use_predefined_hps=True,  # Use predefined hyperparameter space
        trial_num_threads=4,  # Number of threads to train models in each trial
        trial_maximum_training_duration_seconds=600  # Maximum training time per trial in seconds
    )

    # Initialize the Gradient Boosted Trees model with the tuner
    model = tfdf.keras.GradientBoostedTreesModel(
        task=tfdf.keras.Task.CLASSIFICATION,
        tuner=tuner,
        random_seed=42
    )

    # Prepare the dataset
    train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(inputs, label="label")
    model.fit(train_ds)

    return model