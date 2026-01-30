#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.4.2/icetray-start
#METAPROJECT icetray/v1.17.0
"""
Created on Thu Jun 28 13:33:40 2018

@author: fmcnally & Ethan Dorr ... and jstowers!!!

Extracts IceTop hit data to prepare for CR CNN

Previous cvmfs: /cvmfs/icecube.opensciencegrid.org/py2-v2/icetray-start
icerec/V05-01-06
"""

import argparse
from pathlib import Path

import numpy as np

from icecube import icetray
from icecube.icetray import I3Tray
from icecube.frame_object_diff.segments import uncompress



def get_omkey(om):
    return '%02i%02i' % (om.string, om.om)


# Find the maximum charge pulse for a particular DOM that does not have a NaN or inf value associated with it
def find_max_q(om, pulses):
    max_q, max_idx = float('-inf'), None
    for i, pulse in enumerate(pulses):
        if pulse.charge > max_q and np.isfinite(pulse.charge) and np.isfinite(pulse.time):
            max_q, max_idx = pulse.charge, i
    return max_idx


class extraction(icetray.I3ConditionalModule):
    def __init__(self, context):
        icetray.I3ConditionalModule.__init__(self, context)

        # Detector parameters
        self.AddParameter('position',        'position of tanks', None)
        self.AddParameter('gain',            'high/low gain of DOM', None)

        # Event-specific parameters
        self.AddParameter('charge_HLC',      'HLC charge deposit values', None)
        self.AddParameter('charge_SLC',      'SLC charge deposit values', None)
        self.AddParameter('time_HLC',        'HLC time deposit values', None)
        self.AddParameter('time_SLC',        'SLC time deposit values', None)
        self.AddParameter('file_info',       'run ID and event ID', None)
        self.AddParameter('energy',          'energy of primary', None)
        self.AddParameter('comp',            'composition of primary', None)
        self.AddParameter('dir',             'direction of primary', None)
        self.AddParameter('plane_dir',       'direction of plane fit', None)
        self.AddParameter('laputop_dir',     'direction of laputop fit', None)
        self.AddParameter('small_dir',       'direction of laputop small fit', None)

        """ CODE GOES HERE """

        # Cuts
        self.AddParameter('passed_STA5',     'whether the event passes the STA5 filter', None)
        self.AddParameter('uncontained_cut', 'whether the event passes the uncontained cut', None)
        self.AddParameter('quality_cut',     'whether the event passes the quality cut', None)


    def Configure(self):
        # Detector parameters
        self.position        = self.GetParameter('position')
        self.gain            = self.GetParameter('gain')
        # Event-specific parameters
        self.charge_HLC      = self.GetParameter('charge_HLC')
        self.charge_SLC      = self.GetParameter('charge_SLC')
        self.time_HLC        = self.GetParameter('time_HLC')
        self.time_SLC        = self.GetParameter('time_SLC')
        self.file_info       = self.GetParameter('file_info')
        self.energy          = self.GetParameter('energy')
        self.comp            = self.GetParameter('comp')
        self.direction       = self.GetParameter('dir')
        self.plane_dir       = self.GetParameter('plane_dir')
        self.laputop_dir     = self.GetParameter('laputop_dir')
        self.small_dir       = self.GetParameter('small_dir')

        """ CODE GOES HERE """

        # Cuts
        self.passed_STA5     = self.GetParameter('passed_STA5')
        self.uncontained_cut = self.GetParameter('uncontained_cut')
        self.quality_cut     = self.GetParameter('quality_cut')


    def Geometry(self, frame):
        self.geometry = frame['I3Geometry']
        self.PushFrame(frame)


    def Calibration(self, frame):
        self.calibration = frame['I3Calibration']
        self.PushFrame(frame)


    def DetectorStatus(self, frame):
        status = frame['I3DetectorStatus']
        # Extract info on tank positions
        self.omList = []
        tankx, tanky, gain = {},{},{}

        # Build station geometry, including gain
        for station_id in self.geometry.stationgeo:
            station = self.geometry.stationgeo[station_id]
            for tank in station:
                for om in tank.omkey_list:
                    omstr = get_omkey(om)
                    # Code fails at (39,61) (DOM not functional?)
                    # ^ Well, it does show up in the Bad DOMs list
                    try: gain[omstr] = str(status.dom_status[om].dom_gain_type)
                    except KeyError: 
                        continue
                    self.omList += [omstr]
                    tankx[omstr] = tank.position.x
                    tanky[omstr] = tank.position.y

        self.gain += [gain]
        self.position += [tankx, tanky]
        self.PushFrame(frame)


    def Physics(self, frame):
        # Must have 3 hits
        if not frame['QFilterMask']['IceTopSTA3_12'].condition_passed:
            self.PushFrame(frame)
            return

        data = [[{}, {}], [{}, {}]]
        pulse_frames = ['OfflineIceTopHLCTankPulses', 'OfflineIceTopSLCTankPulses']
        for i, (data_pair, pulse_frame) in enumerate(zip(data, pulse_frames)):
            pulse_map = frame[pulse_frame]
            for om, pulses in pulse_map.items():
                pulse_max_index = find_max_q(om, pulses)
                if pulse_max_index is None:
                    continue
                pulse = pulses[pulse_max_index]
                omstr = get_omkey(om)
                data_pair[0][omstr] = pulse.charge
                data_pair[1][omstr] = pulse.time

        atts = [[self.charge_HLC, self.time_HLC], [self.charge_SLC, self.time_SLC]]
        for att_pair, data_pair in zip(atts, data):
            for att, val in zip(att_pair, data_pair):
                att.append(val)

        # Event identification
        self.event_header = frame['I3EventHeader']
        RunID = self.event_header.run_id
        EventID = self.event_header.event_id
        ID = '%06i_%06i' % (RunID, EventID)
        self.file_info.append(ID)

        # Primary information
        primary = frame['MCPrimary']
        self.energy += [primary.total_energy]
        self.comp += [primary.type_string]
        self.direction += [[primary.dir.theta, primary.dir.phi]]

        plane_reco = [None, None]
        if 'ShowerPlane' in frame.keys():
            planeFit = frame['ShowerPlane']
            if planeFit.fit_status.name == 'OK':
                plane_reco = [planeFit.dir.theta, planeFit.dir.phi]
        self.plane_dir.append(plane_reco)

        lap_reco = [None, None]
        small_reco = [None, None]
        if 'Laputop' in frame.keys():
            laputopFit = frame['Laputop']
            check = laputopFit.fit_status.name == 'OK'
            if check:
                lap_reco = [laputopFit.dir.theta, laputopFit.dir.phi]
            # Additionally extract small shower reco if Laputop fails
            smallFit = laputopFit if check else frame['LaputopSmall']
            if smallFit.fit_status.name == 'OK':
                small_reco = [smallFit.dir.theta, smallFit.dir.phi]
        self.laputop_dir.append(lap_reco)
        self.small_dir.append(small_reco)

        """ CODE GOES HERE """

        # Data/Quality cuts
        self.passed_STA5 += [frame['QFilterMask']['IceTopSTA5_12'].condition_passed]
        self.uncontained_cut += [frame['IT73AnalysisIceTopQualityCuts']['IceTop_StandardFilter'] and not frame['IT73AnalysisIceTopQualityCuts']['IceTopMaxSignalInside']]
        self.quality_cut += [all(frame['IT73AnalysisIceTopQualityCuts'].values()) and np.pi - frame['Laputop'].dir.theta <= 40 * np.pi/180]

        self.PushFrame(frame)
        return


if __name__ == "__main__":

    l3sim = '/data/ana/CosmicRay/IceTop_level3/sim/IC86.2012'
    default_gcd = '%s/GCD/Level3_12360_GCD.i3.gz' % l3sim
    default_file = '%s/oldstructure/12360/Level3_IC86.2012_12360_Run012079.i3.gz' % l3sim

    parser = argparse.ArgumentParser(
        description='Extracts information for use with CR CNN energy reco')
    parser.add_argument('-g', '--gcd', dest='gcdFile', 
        default=default_gcd)
    parser.add_argument('-f', '--files', dest='files', nargs='*',
        default=[default_file],
        help='List of input files to run over')
    parser.add_argument('-o', '--out', dest='out',
        help='Output file name')
    args = parser.parse_args()

    args.files.insert(0, args.gcdFile) 

    kwargs = ['position', 'gain', # Detector parameters
              'charge_HLC', 'charge_SLC', 'time_HLC', 'time_SLC', # DOM data
              'file_info', 'energy', 'comp', 'dir', 'plane_dir', 'laputop_dir', 'small_dir', # Event info

              # Code goes here #

              'passed_STA5', 'uncontained_cut', 'quality_cut'] # Cuts
    data = {k:[] for k in kwargs}

    tray = I3Tray()
    tray.AddModule('I3Reader','reader', FileNameList=args.files)
    tray.AddSegment(uncompress, 'uncompress')
    tray.AddModule(extraction, **data)
    tray.Execute()

    np.save(args.out, data)
