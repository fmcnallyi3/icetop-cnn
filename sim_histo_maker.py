import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from data_tools import r_log


##############################################################################################

# ETHAN'S SIMULATION DATA HISTOGRAM MAKER
# REQUIRES ACCESS TO DATA_TOOLS AND SIMDATA

""" \/ FEEL FREE TO CHANGE \/ """

# COMP YOU WISH TO ANALYZE
comp = ['p','h','o','f']
# LOCATION OF SIMDATA
simdata_prefix = os.getcwd() + '/simdata'
# DEFAULT OUT DIRECTORY
# WILL BE CREATED AUTOMATICALLY BY DEFAULT IF NOT ALREADY CREATED
out_dir = os.getcwd() + '/histograms/sim'

##############################################################################################


def setup():

    x_files = sorted(glob('%s/x_*.npy' % simdata_prefix))
    comp_dict = {'p':'12360','h':'12630','o':'12631','f':'12362'}
    sim_list = [comp_dict[c] for c in comp]

    x_files = [f for f in x_files if any([sim in f for sim in sim_list])]

    # Create bins
    # DEFAULT BINS ARE THE SAME AS USED FOR REAL DATA TO ENABLE DIRECT COMPARISON
    bins = [np.insert(np.logspace(-5, 4, 300, base=10), 0, 0.), np.logspace(3.75, 7.25, 301, base=10)]
    return x_files, bins


def load_data(x_files, bins):

    # [Charge, Time]
    # Create blank histograms, divide by HLC/SLC
    base_hists_hlc = [np.zeros(300, dtype=float), np.zeros(300, dtype=float)]
    base_hists_slc = [np.zeros(300, dtype=float), np.zeros(300, dtype=float)]

    for i, xf in enumerate(x_files):
        print('Working on file %i/%i...' % ((i+1), len(x_files)), end='\r' )
        x_i = np.load(xf)

        qhlc, thlc = r_log(x_i[...,::2][...,:2]), x_i[...,::2][...,2:]
        qslc, tslc = r_log(x_i[...,1::2][...,:2]), x_i[...,1::2][...,2:]

        # HLC charge
        base_hists_hlc[0] += np.histogram(qhlc, bins=bins[0])[0]
        # HLC time
        base_hists_hlc[1] += np.histogram(thlc, bins=bins[1])[0]
        # SLC charge
        base_hists_slc[0] += np.histogram(qslc, bins=bins[0])[0]
        # SLC time
        base_hists_slc[1] += np.histogram(tslc, bins=bins[1])[0]
    print()

    return base_hists_hlc, base_hists_slc


def create_histograms(base_hists_hlc, base_hists_slc, bins):

    names = ['charge_temp', 'time_temp']
    LC_types = ['HLC', 'SLC', 'CLC']
    val_types = ['Charge', 'Time']
    units = ['[VEM]', '[ns]']
    nuclei = ''.join(comp)

    # Iterate through charge and time in HLC and SLC simultaneously using zip()
    for i, (HLC_hist, SLC_hist) in enumerate(zip(base_hists_hlc, base_hists_slc)): 
        # Need a HLC, SLC, and CLC histogram
        hists = [HLC_hist, SLC_hist, HLC_hist + SLC_hist]
        # Iterate through different histogram types
        for j, LC_type in enumerate(LC_types):
            _, ax = plt.subplots()
            _, _, _ = ax.hist(bins[i%2][:-1], bins[i%2], weights=hists[j])
            plt.title(f'Simulation Data 2012 {nuclei.upper()} {LC_type} {val_types[i%2]} Histogram')
            plt.xlabel(f'{val_types[i%2]} {units[i%2]}')
            plt.ylabel('Count')
            ax.set_xscale('log')
            ax.set_yscale('log')

            # Automatically create output directories if not specified
            try:
                plt.savefig('%s/%s/%s' % (out_dir, LC_type, names[i].replace('temp', LC_type) + '_%s.png' % nuclei))
            except FileNotFoundError:
                for LC_dir in LC_types:
                    os.makedirs(f'{out_dir}/{LC_dir}')
                plt.savefig('%s/%s/%s' % (out_dir, LC_type, names[i].replace('temp', LC_type) + '_%s.png' % nuclei))

            plt.clf()


def main():

    print('Loading setup...')
    x_files, bins = setup()

    print('Loading data...')
    base_hists_hlc, base_hists_slc = load_data(x_files, bins)

    print('Creating histograms...')
    create_histograms(base_hists_hlc, base_hists_slc, bins)

    print('Done!')


if __name__ == '__main__':
    main()
