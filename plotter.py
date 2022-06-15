#!/usr/bin/env python

import glob
import argparse
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

from collections import defaultdict

from data_tools import modLog, hexSquare


## Pair two tanks within a station into a single object at average position
def pairXY(doms, xcoords, ycoords, nn=15, deepCore=False):

    dom2xy = {}
    nn = 15  # maximum distance to nearest neighbor tank
    xcoords = xcoords.astype('float')
    ycoords = ycoords.astype('float')

    for dom, x_i, y_i in zip(doms, xcoords, ycoords):
        xDists = abs(x_i - xcoords)
        yDists = abs(y_i - ycoords)
        distCut = (xDists < nn) * (yDists < nn)
        x_ave = np.mean(xcoords[distCut])
        y_ave = np.mean(ycoords[distCut])
        dom2xy[dom] = [x_ave, y_ave]

    # Remove infill tanks -- not sure if this is necessary/desirable
    # Mirco does this by treating infill stations as separate array
    if not deepCore:
        dom2xy = {k:v for k, v in dom2xy.items() if int(k[:2]) < 79}

    return dom2xy


# Simplified version that moves between station number and position
def stationXY(dom2xy):
    sta2xy = {}
    for key in dom2xy.keys():
        if key[:2] not in sta2xy:
            sta2xy[key[:2]] = dom2xy[key]
    return sta2xy


if __name__ == "__main__":

    p = argparse.ArgumentParser(
            description='Converts distribution of charges into square lattice')
    p.add_argument('--text', dest='text',
            default=False, action='store_true',
            help='Display station numbering on images')
    p.add_argument('--event', dest='event', 
            type=int, default=0,
            help='Optional event number')
    args = p.parse_args()

    files = sorted(glob.glob('simFiles/*.npy'))
    f = files[0]

    # Extract information from simulation
    d = np.load(f, allow_pickle=True)
    d = d.item()
    x, y = d['position']
    charges = d['charge'][args.event]
    times = d['time'][args.event]
    # Remove infill array
    charges = {k:v for k, v in charges.items() if int(k[:2])<79}
    times = {k:v for k, v in times.items() if int(k[:2])<79}

    # Generate dictionaries for doms, stations, and xy positions
    doms, xValues = np.transpose([[dom, xValue] for dom, xValue in x.items()])
    doms, yValues = np.transpose([[dom, yValue] for dom, yValue in y.items()])
    dom2xy = pairXY(doms, xValues, yValues)
    sta2xy = stationXY(dom2xy)
    stationList = sta2xy.keys()

    # Figure setup
    fig = plt.figure(figsize=(12,12))
    grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)

    # Plot charges in true layout
    tankX = [x[dom] for dom in charges.keys()]
    tankY = [y[dom] for dom in charges.keys()]
    # Optional average position of station
    newX, newY = np.transpose([sta2xy[s] for s in stationList])
    # Marker size corresponds to charge
    r = modLog(np.array([charges[dom] for dom in charges.keys()]))
    r = (20 * r/r.max())**2 # 0 to 10 point radii(?)
    # Color corresponds to arrival time
    c = np.array([times[dom] for dom in charges.keys()])
    c = (c - c.min()) / (c.max()-c.min())

    ax1 = plt.subplot(grid[0,0])
    ax1.scatter(xValues.astype('float'), yValues.astype('float'),
            c='k', marker='.', s=0.1)
    ax1.scatter(tankX, tankY, s=r, c=c, alpha=0.3, cmap='rainbow_r')
    # ax1.scatter(newX, newY)
    if args.text:
        for i, txt in enumerate(stationList):
            ax1.annotate(txt, (newX[i], newY[i]))

    # Square lattice representation
    ## NOTE: this seems to be working, but is inverted in terms of x&y
    ## due to Mirco's conventions. Be careful with this moving forward
    ax2 = plt.subplot(grid[1,0])
    sta2hex, hex2sta = hexSquare()
    # Station locations
    squareY, squareX = np.transpose([sta2hex[s] for s in stationList])
    ax2.scatter(squareX, squareY, c='k', marker='.', s=0.1)
    # Sum charges and average times
    staCharges = defaultdict(float)
    for dom, charge in charges.items():
        staCharges[dom[:2]] += charge
    tempTimes, staTimes = defaultdict(list), defaultdict(float)
    for dom, time in times.items():
        tempTimes[dom[:2]] += [time]
    for s, timeList in tempTimes.items():
        staTimes[s] = np.mean(timeList)
    staY, staX = np.transpose([sta2hex[s] for s in staCharges.keys()])
    r = modLog(np.array([staCharges[s] for s in staCharges.keys()]))
    r = (20 * r/r.max())**2
    c = np.array([staTimes[s] for s in staCharges.keys()])
    c = (c - c.min()) / (c.max()-c.min())
    ax2.scatter(staX, staY, s=r, c=c, alpha=0.5, cmap='rainbow_r')
    if args.text:
        for i, txt in enumerate(stationList):
            ax2.annotate(txt, (squareX[i], squareY[i]))

    # Stacked image representations of charge and time
    nx = squareX.max() - squareX.min() + 1
    ny = squareY.max() - squareY.min() + 1
    chargeMap = np.zeros((ny, nx))
    for s, charge in staCharges.items():
        y_i = sta2hex[s][0] - squareY.min()
        x_i = sta2hex[s][1] - squareX.min()
        chargeMap[y_i][x_i] += charge
    timeMap = np.zeros((ny, nx))
    timeX, timeY = np.zeros(nx), np.zeros(ny)
    for s, time in staTimes.items():
        y_i = sta2hex[s][0] - squareY.min()
        x_i = sta2hex[s][1] - squareX.min()
        timeMap[y_i][x_i] += time

    # Normalize maps, ignoring 0 values
    timeMap -= timeMap[timeMap!=0].min()
    timeMap /= timeMap.max()
    chargeMap -= chargeMap[chargeMap!=0].min()
    chargeMap /= chargeMap.max()

    #im3 = ax3.imshow(chargeMap, origin='lower')
    #ax3_divider = make_axes_locatable(ax3)
    #cax3 = ax3_divider.append_axes("right", size="7%", pad="2%")
    #cb3 = colorbar(im3, cax=cax3)
    #imX = squareX - squareX.min()
    #imY = squareY - squareY.min()
    #if args.text:
    #    for i, txt in enumerate(stationList):
    #        ax3.annotate(txt, (imX[i], imY[i]))

    ax3 = plt.subplot(grid[:,1:], projection='3d')
    ax3.grid(False)
    ax3.axis('off')
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_zticks([])

    tx, ty = np.arange(0,nx+1,1), np.arange(0,ny+1,1)
    tx, ty = np.meshgrid(tx, ty)
    Z = np.zeros_like(tx)
    my_cmap = cm.rainbow_r
    my_cmap.set_under('gray', alpha=0.05)
    my_cmap2 = cm.viridis
    my_cmap2.set_under('gray', alpha=0.05)

    ax3.plot_surface(tx, Z+0.1, ty, rstride=1, cstride=1,
            facecolors=my_cmap(timeMap), shade=False)
    ax3.plot_surface(tx, Z, ty, rstride=1, cstride=1,
            facecolors=my_cmap2(chargeMap), shade=False)

    #im4 = ax3.imshow(timeMap, origin='lower', cmap=my_cmap, clim=[tMin,tMax])
    #ax3_divider = make_axes_locatable(ax3)
    #cax3 = ax3_divider.append_axes("right", size="7%", pad="2%")
    #cb4 = colorbar(im4, cax=cax3)
    #imX = squareX - squareX.min()
    #imY = squareY - squareY.min()
    #if args.text:
    #    for i, txt in enumerate(stationList):
    #        ax3.annotate(txt, (imX[i], imY[i]))

    plt.show()


