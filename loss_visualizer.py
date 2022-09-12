# LOSS VISULARIZER
# WRITTEN BY ETHAN DORR
# REQUIRES 'images' FOLDER IN WORKING DIRECTORY

import csv
import matplotlib.pyplot as plt

# ADD MODEL NAMES TO ANALYZE AS A LIST OF STRINGS
model_names = []

labels = ['huber', 'mae', 'mse']
for model_name in model_names:
    epochs = []
    training_losses = [[], [], []]
    validation_losses = [[], [], []]

    with open(f'models/{model_name}', 'r') as model_file:
        rows = csv.reader(model_file, delimiter=',')
        for i, row in enumerate(rows):
            if i == 0:
                continue
            epochs.append(int(i-1))
            for j, (training_loss, validation_loss) in enumerate(zip(training_losses, validation_losses)):
                training_loss.append(float(row[j+1]))
                validation_loss.append(float(row[j+4]))

    fig, axs = plt.subplots(3)
    fig.suptitle(model_name)
    for i, (training_loss, validation_loss) in enumerate(zip(training_losses, validation_losses)):
        axs[i].plot(epochs, training_loss, label=f'{labels[i]}_training_loss')
        axs[i].plot(epochs, validation_loss, label=f'{labels[i]}_validation_loss')
        axs[i].legend()
        axs[i].set_yscale('log')
    plt.savefig(f'images/{model_name}.png', format='png')


    

