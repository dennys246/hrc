import mne, scipy, random
from glob import glob
from itertools import compress
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def load_pcat(bids_dir, ex_subs = [], deconvolution = False):
    # make a list where all of the scans will get loaded into
    subject_ids = []
    raw_scans = []
    preproc_scans = []

    # Load in master file with scan order info
    subject_dirs = glob(f'{bids_dir}*/')
    print(subject_dirs)

    for dir_ind, directory in enumerate(subject_dirs):
        for excluded in ex_subs:
            if excluded in directory:
                print(f"Deleting {directory}")
                del subject_dirs[dir_ind]

    scan_events = []
    for subject_dir in subject_dirs:

        snirf_files = glob(f"{subject_dir}/*_Flanker/*.snirf")
        if len(snirf_files) == 0:
            continue
        
        subject_ids.append(subject_dir.split('/')[-2])

        flanker_dir = '/'.join(snirf_files[0].split('/')[:-1])
        raw_nirx = mne.io.read_raw_snirf(snirf_files[0])
        
        # Grab events
        events = load_events(flanker_dir)
        print(f"events: {events}")
        if len(events) == 0: # If no events
            print(f"Skipping {subject_dir}, no events found...")
            continue

        raw_scans.append(raw_nirx)
        preproc_scans.append(preprocess(raw_nirx, deconvolution))
        scan_events.append(events)
        
    return subject_ids, raw_scans, preproc_scans, scan_events


def load_care(bids_dir, ex_subs = [], deconvolution = False):
    # make a list where all of the scans will get loaded into
    subject_ids = []
    child_scans = []
    parent_scans = []

    # Load in master file with scan order info
    subject_dirs = glob(f'{bids_dir}*/')
    print(subject_dirs)

    for dir_ind, directory in enumerate(subject_dirs):
        for excluded in ex_subs:
            if excluded in directory:
                print(f"Deleting {directory}")
                del subject_dirs[dir_ind]

    for subject_dir in [subject_dirs[random.randint(0, len(subject_dirs) - 1)] for ind in range(0, 25)]:

        probe_files = glob(f"{subject_dir}/*/*/*_probeInfo.mat")
        
        if len(probe_files) == 0:
                print("No probes found...")
                continue
        
        for probe_file in probe_files:
            
            probe_split = probe_file.split('/')

            subject_ids.append(subject_dir.split('/')[-3])

            nirs_folder = "/".join(probe_split[:-1])
            main_folder = probe_split[-2]

            raw_nirx = mne.io.read_raw_nirx(nirs_folder)

            if len(main_folder) == 13: # If parent ID
                parent_scans.append(preprocess(raw_nirx, deconvolution))
            else: # If child ID
                child_scans.append(preprocess(raw_nirx, deconvolution))
    print(parent_scans)
    return subject_ids, child_scans, parent_scans

def load_events(dir):
    # header =  ['sample', '1', 'block', 'directionality', 'congruency', 'direction', 'response', 'accuracy']
    # Grab events
    event_files = glob(f"{dir}/*.evt")
    print(event_files)
    for event_file in event_files:
        if event_file[-8:] == '_old.evt':
            continue
        return pd.read_csv(event_file, sep = '\t', header = None, names = ['sample', '0', '1', '2', '3', '4', '5', '6', '7', '8'])


def process_congruency(events):
    """ Process a series of events for P-CAT Flanker and extract congruency """
    event_id = {
        'incongruent' : 1,
        'congruent' : 2
    }
    new_events = []
    for ind, trial in events.iterrows():
        # Handle no direction case
        if int(trial['2']) == 0:
            #new_events.append([sample, 0, 3]) # Append on a event for non-directional
            continue # exclude non-directional trials
        else:
            sample, event = int(trial['sample']), int(trial['3'])
            event += 1 # Adjust id to match event ID
            new_events.append([sample, 0 , event])
    return pd.array(new_events), event_id

def process_accuracy(events):
    """ Process a series of events for P-CAT Flanker to extract whether an answer was correct or incorrect """
    event_id = {
        'incorrect' : 1,
        'correct' : 2
    }
    new_events = []
    for ind, trial in events.iterrows():
        # Handle no direction case
        if int(trial['5']): # If response given (i.e. 1)
            sample, event = int(trial['sample']), int(trial['6'])
            event += 1 # Adjust id to match event ID
            new_events.append([sample, 0 , event])
    return pd.array(new_events), event_id


def preprocess(scan, deconvolution = False):
    """
    Preprocess fNIRS data in scan mne object passed in.
    """

    #try:
    # convert to optical density
    scan.load_data() 
    data, times = scan.get_data(return_times=True)

    raw_od = mne.preprocessing.nirs.optical_density(scan)

    # scalp coupling index
    sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)
    raw_od.info['bads'] = list(compress(raw_od.ch_names, sci < 0.5))

    if len(raw_od.info['bads']) > 0:
        print("Bad channels in subject", raw_od.info['subject_info']['his_id'], ":", raw_od.info['bads'])

    # temporal derivative distribution repair (motion attempt)
    tddr_od = mne.preprocessing.nirs.tddr(raw_od)

    # This function computes and applies a projection that removes the specified polynomial trend without cutting into the frequency spectrum, so you preserve everything from DC to high frequency — which is ideal when estimating the full HRF shape
    #detrended_od = polynomial_detrend(tddr_od, order=2)

    # haemoglobin conversion using Beer Lambert Law (this will change channel names from frequency to hemo or deoxy hemo labelling)
    haemo = mne.preprocessing.nirs.beer_lambert_law(tddr_od, ppf=0.1)

    # bandpass filter if not deconvolving the data
    if not deconvolution:
        haemo_bp =  haemo.copy().filter(0.01, 0.2, h_trans_bandwidth=0.1, l_trans_bandwidth=0.02)
        return haemo_bp
    else:
        return haemo

def polynomial_detrend(raw, order = 1):
    raw_detrended = raw.copy()
    times = raw.times
    X = np.vander(times, N = order + 1, increasing = True)  # Design matrix for polynomial trend

    for idx in range(len(raw.ch_names)):
        y = raw.get_data(picks = idx)[0]
        beta = np.linalg.lstsq(X.T @ X, X.T @ y, rcond = None)[0]
        y_detrended = y - X @ beta
        raw_detrended._data[idx] = y_detrended

    return raw_detrended