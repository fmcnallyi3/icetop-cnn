import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from glob import glob

##############################################################################################

# ETHAN DORR'S SIMULATION DATA HISTOGRAM MAKER
# REQUIRES ACCESS TO SIMDATA

# LOCATION OF SIMDATA
simdata_prefix = os.getcwd() + '/simdata'

##############################################################################################


def setup():

    x_files = sorted(glob('%s/x_*.npy' % simdata_prefix))
    comp_dict = {'p':'12360','h':'12630','o':'12631','f':'12362'}
    sim_list = [comp_dict[c] for c in args.comp]

    x_files = [f for f in x_files if any([sim in f for sim in sim_list])]

    # Create bins
    if args.is_data_comparison:
        # BINS FOR COMPARING REAL DATA
        if args.is_linear:
            bins = [np.linspace(0.00001, 10000, 301), np.linspace(5600, 17782800, 301)]
        else:
            bins = [np.logspace(-5, 4, 301, base=10), np.logspace(3.75, 7.25, 301, base=10)]
    else:
        # BINS FOR ANALYZING SIMULATION DATA
        if args.is_linear:
            bins = [np.linspace(0.007, 10000, num=301), np.linspace(7000, 22400, num=301)]
        else:
            bins = [np.logspace(-2.15, 4, 301, base=10), np.logspace(3.85, 4.35, 301, base=10)]

    return x_files, bins


def load_data(x_files, bins):

    # [Charge, Time]
    # Create blank histograms, divide by HLC/SLC
    base_hists_hlc = [np.zeros(300, dtype=int), np.zeros(300, dtype=int)]
    base_hists_slc = [np.zeros(300, dtype=int), np.zeros(300, dtype=int)]

    for i, xf in enumerate(x_files):
        print('Working on file %i/%i...' % ((i+1), len(x_files)), end='\r' )
        x_i = np.load(xf)

        # Perform r_log on hlc charge - ripped from data_tools
        qhlc = np.sign(x_i[...,::2][...,:2])*(10**np.abs(x_i[...,::2][...,:2]) - 1)
        # hlc time
        thlc = x_i[...,::2][...,2:]
        # Perform r_log on slc charge - ripped from data_tools
        qslc = np.sign(x_i[...,1::2][...,:2])*(10**np.abs(x_i[...,1::2][...,:2]) - 1)
        # slc time
        tslc = x_i[...,1::2][...,2:]

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


def get_clip_index(hist_clc, clip_percent):

    num_pulses, counter = sum(hist_clc), 0
    
    # Quits on first loop for clip_percent < 0
    for i, curr_bin in enumerate(reversed(hist_clc)):
        counter += curr_bin
        if counter > num_pulses * clip_percent:
            return len(hist_clc) - i - 1
    # clip_percent >= 1
    return np.nonzero(hist_clc)[0][0]
    

def clip_histo(hist, clip_idx):

    sum_bins = sum(hist[clip_idx:])
    hist[clip_idx] = sum_bins
    hist[clip_idx+1:] = 0
    return hist


def create_histograms(base_hists_hlc, base_hists_slc, bins):

    names = ['charge_temp', 'time_temp']
    if args.is_cumulative:
        names = [f'cumul_{name}' for name in names]
    clip_percents = [args.q_clip_percent, args.t_clip_percent]
    LC_types = ['HLC', 'SLC', 'CLC']
    val_types = ['Charge', 'Time']
    units = ['[VEM]', '[ns]']

    # Iterate through charge and time in HLC and SLC simultaneously using zip()
    for i, (HLC_hist, SLC_hist) in enumerate(zip(base_hists_hlc, base_hists_slc)): 
        # Need a HLC, SLC, and CLC histogram
        hists = [HLC_hist, SLC_hist, HLC_hist + SLC_hist]
        # Find the right bin index to clip into based on the CLC histogram
        clip_idx = get_clip_index(hists[2], clip_percents[i])
        print(f'{val_types[i]} Clip Range: {bins[i][clip_idx]:.2f} - {bins[i][clip_idx+1]:.2f} {units[i]}')
        # Clip all histograms
        hists = [clip_histo(hist, clip_idx) for hist in hists]
        # Make cumulative histograms if prompted
        if args.is_cumulative:
            hists = [np.cumsum(hist[::-1])[::-1] for hist in hists]

        # Iterate through different histogram types
        for j, LC_type in enumerate(LC_types):
            _, ax = plt.subplots()
            _, _, _ = ax.hist(bins[i][:-1], bins[i], weights=hists[j])
            plt.title(f'Simulation Data 2012 {args.comp.upper()} {LC_type} {val_types[i]} Histogram')
            plt.xlabel(f'{val_types[i]} {units[i]}')
            plt.ylabel('Count')
            plt.legend([f'q_clip : {args.q_clip_percent}   t_clip : {args.t_clip_percent}'])

            if not args.is_linear:
                ax.set_xscale('log')
            ax.set_yscale('log')

            # Automatically create output directories
            name = names[i].replace('temp', LC_type) + f'_{args.comp}_q{args.q_clip_percent}_t{args.t_clip_percent}_'

            plot_num = 0
            while(os.path.exists(f'{args.out_dir}/{LC_type}/{name+str(plot_num)+".png"}')): plot_num += 1
            name += str(plot_num)

            try:
                plt.savefig(f'{args.out_dir}/{LC_type}/{name}.png')
            except FileNotFoundError:
                for LC_dir in LC_types:
                    os.makedirs(f'{args.out_dir}/{LC_dir}', exist_ok=True)
                plt.savefig(f'{args.out_dir}/{LC_type}/{name}.png')

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

    parser = argparse.ArgumentParser(
            description='Makes clipped histograms of charge and time')
    parser.add_argument('--comp', dest='comp', type=str,
            default='phof',
            help='Composition(s) to make histograms from')
    parser.add_argument('-c', '--cumulative', dest='is_cumulative',
            default=False, action='store_true',
            help='Option to create a reversed cumulative histogram')
    parser.add_argument('-d', '--data', dest='is_data_comparison',
            default=False, action='store_true',
            help='Option for unzooming histograms to fit view for real data')
    parser.add_argument('-l', '--linear', dest='is_linear',
            default=False, action='store_true',
            help='Option for creating histogram with linear axes/bins')
    parser.add_argument('-o', '--out', dest='out_dir', type=str,
            default=f'{os.getcwd()}/histograms/sim',
            help='Output file path')
    parser.add_argument('-q', '--charge', dest='q_clip_percent', type=float,
            default=0.0,
            help='Percentage of pulses to clip in charge, range [0,1]')
    parser.add_argument('-t', '--time', dest='t_clip_percent', type=float,
            default=0.0,
            help='Percentage of pulses to clip in time, range [0,1]')
    args = parser.parse_args()

    main()
