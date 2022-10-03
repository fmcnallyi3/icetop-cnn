from glob import glob
from data_tools import data_prep, get_cut, get_data_cut, load_preprocessed
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import os


# Edit this file path to the models folder containing .h5 and .npy files for each model.
model_prefix = os.getcwd()+'/models'

# Edit this file path to the folder containing the simulation data.
sim_prefix = os.getcwd()+'/simdata'

# Option to change font size for all labels within this notebook
label_params = {'fontsize':16}


# List of available models

model_list = sorted(glob('%s/*.h5' % model_prefix))
model_list = [os.path.basename(m)[:-3] for m in model_list]

param_list = sorted(glob('%s/*.npy' % model_prefix))
param_list = [os.path.basename(p)[:-4] for p in param_list]

print('Available models:', sorted(set(model_list).intersection(param_list)))
print('\nModels without parameter files:', sorted(set(model_list).difference(param_list)))


# Keys you want to study
key_list = []
descriptions = []

# Automatic intake of parameters from parameter files
labels, p = {}, {}
for key, description in zip(key_list, descriptions):
    labels[key] = description
    d = np.load('%s/%s.npy' % (model_prefix, key), allow_pickle=True)
    p[key] = d.item()
    print(key, ':', p[key])


# Load data, x in eight layers (q1h, q1s, q2h, q2s, t1h, t1s, t2h, t2s) and y as a dictionary with event-level parameters
x, y = load_preprocessed(sim_prefix, comp=['p','h','o','f'])


models = {}
recoE = {}
data_cut = {}

# Calculate reconstructed energies. This can take a bit, but should print out info on each key as it works
for key in key_list:

    # Comment these two lines if you want to rerun your energy reconstructions each time
    if key in models.keys():
        continue
    
    print('Working on %s...' % key)
    print(p[key])
    # Note: very sensitive to tensorflow/keras version.
    models[key] = load_model('%s/%s.h5' % (model_prefix, key)) # Edit file path
    
    # Configure input data
    x_i, idx, pre_cut = data_prep(x, y, 'assessment', **p[key]) # returns cut x_i, index of where time begins, and the train/assess + sta5 cut

    # Cut data and filter NaN's if trained on a zenith reconstruction
    cut_for_data = get_data_cut(p[key]['reco'], y, pre_cut)
    
    if p[key]['reco'] != None:
        x_i[0] = x_i[0][cut_for_data] # q/t
        x_i[1] = x_i[1][cut_for_data] # z
    else:
        x_i = x_i[cut_for_data] # q/t
    
    # Save data cut
    data_cut[key] = cut_for_data

    # Models should only output energy
    if p[key]['reco'] != None:
        recoE[key] = models[key].predict([x_i[0], x_i[1]]).flatten()
    else:
        recoE[key] = models[key].predict(x_i).flatten()

    print()


ebins = np.linspace(5, 8, 181)
evalues = (ebins[:-1] + ebins[1:]) / 2

cut_names = ['No Cut', 'Quality Cut']
ncols, nrows = len(cut_names), len(key_list)

# Create output directory if necessary
if not os.path.isdir(f'{os.getcwd()}/assessment'):
    os.mkdir(f'{os.getcwd()}/assessment')

# Plot logged energy resolution

hist_args = {'range':(-2,2), 'bins':121, 'density':True, 'histtype':'step', 'log':True, 'linewidth':4}
fig, axs = plt.subplots(figsize=(13*ncols, 8), ncols=ncols)

for i, cut_name in enumerate(cut_names):
    ax = axs[i]
    for j, key in enumerate(key_list):
        cut, energy = get_cut(cut_name, y, p[key]['reco'], recoE[key], data_cut[key])
        ax.hist((recoE[key][cut] - energy), label=labels[key], **hist_args)
    ax.set_title('Energy Resolution (%s)' % cut_name, **label_params)
    ax.set_xlabel(r'$\log_{10}(E_{\mathrm{reco}}/\mathrm{GeV}) - \log_{10}(E_{\mathrm{true}}/\mathrm{GeV})$', **label_params)
    ax.set_ylabel('Counts', **label_params)
    ax.legend()
    ax.axvline()
plt.savefig(f'assessment/logged_energy_resolution.png', format='png')


# Plot zoomed and unlogged energy resolution

hist_args = {'range':(-1,1), 'bins':121, 'density':True, 'histtype':'step', 'linewidth':4}
fig, axs = plt.subplots(figsize=(13*ncols, 8), ncols=ncols)

for i, cut_name in enumerate(cut_names):
    ax = axs[i]
    for j, key in enumerate(key_list):
        cut, energy = get_cut(cut_name, y, p[key]['reco'], recoE[key], data_cut[key])
        ax.hist((recoE[key][cut] - energy), label=labels[key], **hist_args)
    ax.set_title('Energy Resolution (%s)' % cut_name, **label_params)
    ax.set_xlabel(r'$\log_{10}(E_{\mathrm{reco}}/\mathrm{GeV}) - \log_{10}(E_{\mathrm{true}}/\mathrm{GeV})$', **label_params)
    ax.set_ylabel('Counts', **label_params)
    ax.legend()
    ax.axvline()
plt.savefig(f'assessment/zoomed_unlogged_energy_resolution.png', format='png')


# Summary parameters

for key in key_list:
    for i, cut_name in enumerate(cut_names):
        cut, energy = get_cut(cut_name, y, p[key]['reco'], recoE[key], data_cut[key])
        median, err_min, err_max = np.percentile(recoE[key][cut] - energy, (50,16,84))
        print('Energy resolution for %s (%s): %.03f +%.03f %.03f' % (key, cut_name, median, err_max, err_min))
    print()


# Plot two-dimensional visualization

np.seterr(divide = 'ignore')

fig, axs = plt.subplots(figsize=(13*ncols, 10*nrows), ncols=ncols, nrows=nrows, 
                        sharex=True, sharey=True)

for i, key in enumerate(key_list):
    for j, cut_name in enumerate(cut_names):
        
        ax = axs[i, j] if len(key_list) > 1 else axs[j]
        cut, energy = get_cut(cut_name, y, p[key]['reco'], recoE[key], data_cut[key])
        
        h, xedges, yedges = np.histogram2d(recoE[key][cut], energy, bins=(ebins, ebins), normed=False, weights=None)
        # Normalize
        ntot = np.sum(h, axis=0).astype(float)
        ntot[ntot==0] = 1.
        h /= ntot
        
        # Create contours
        contour_values = [0.025, 0.16, 0.84, 0.975]
        contour_list = [[] for _ in contour_values]
        for c, col in enumerate(h.transpose()):
            ccol = col.cumsum()
            for l, val in zip(contour_list, contour_values):
                try: l += [np.where(ccol > val)[0][0]]
                except IndexError:
                    l += [0]
        for l in contour_list:
            l.insert(0, l[0])
            if i >= len(contour_list) / 2:
                l = [j+1 for j in l]     
        ax.plot(evalues, evalues, 'k', ls=':')
        for l in contour_list:
            ax.step(ebins, ebins[l], color='red', linestyle='--')
        
        # Plot on a log scale
        extent = [yedges[0], yedges[-1], xedges[0], xedges[-1]]
        im = ax.imshow(np.log10(h), extent=extent, origin='lower', interpolation='none', vmin=-3.5, vmax=-0.5)
        ax.set_title('%s (%s)' % (key, cut_name), **label_params)
        ax.set_xlabel(r'$\log_{10}(E_{\mathrm{true}}/\mathrm{GeV})$', **label_params)
        ax.set_ylabel(r'$\log_{10}(E_{\mathrm{reco}}/\mathrm{GeV})$', **label_params)
        fig.colorbar(im, ax=ax)
plt.savefig(f'assessment/2d_visualization.png', format='png')


# Plot energy resolution as a function of zenith

coszbins = np.linspace(0.4,1,20)
coszvalues = (coszbins[1:]+coszbins[:-1])/2
kwargs = {'fmt':'.',
          'markersize':16,
          'elinewidth':2,
          'capsize':10,
          'capthick':2}
fig, axs = plt.subplots(figsize=(13*ncols, 10), ncols=ncols)

for i, cut_name in enumerate(cut_names):
    ax = axs[i]
    for k, key in enumerate(key_list):
        
        theta = y['laputop_dir'].transpose()[0]
        theta = theta[get_data_cut(p[key]['reco'], y, data_cut[key])]
        theta = np.pi - theta.astype('float')  # Define 0 degrees as overhead

        array_info = np.zeros(shape=(len(coszvalues), 3))
        cut, energy = get_cut(cut_name, y, p[key]['reco'], recoE[key], data_cut[key])
        binned_zenith = np.digitize(np.cos(theta)[cut], coszbins) - 1
        for j in range(len(coszvalues)):
            coszcut = (binned_zenith == j)
            temp_events = recoE[key][cut][coszcut]
            if len(temp_events) != 0:
                array_info[j] = np.percentile(temp_events - energy[coszcut], (50, 16, 84))

        median, err_min, err_max = np.transpose(array_info)
        ax.errorbar(coszvalues, median, yerr=(median-err_min, err_max-median), label=key, **kwargs)
    
    ax.axhline(color='k', ls='--')
    ax.set_title('Energy Resolution v. Zenith (%s)' % cut_name, **label_params)
    ax.set_xlabel(r'$\cos(\theta)$', **label_params)
    ax.set_ylabel(r'$\log_{10}(E_{\mathrm{reco}}/\mathrm{GeV}) - \log_{10}(E_{\mathrm{true}}/\mathrm{GeV})$', **label_params)
    ax.set_ylim(-0.5, 0.5)
    ax.legend() 
plt.savefig(f'assessment/zenith_energy_resolution.png', format='png')
