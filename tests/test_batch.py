import mne, sys, scipy
from glob import glob
from itertools import compress
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sys.path.append("/storage1/fs1/perlmansusan/Active/moochie/github/hrc/hrconv")

import hrc, observer, hrf

def load(bids_dir, ex_subs = []):
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

        subject_ids.append(subject_dir.split('/')[-2])

        snirf_files = glob(f"{subject_dir}/*_Flanker/*.snirf")
        if len(snirf_files) == 0:
            continue

        flanker_dir = '/'.join(snirf_files[0].split('/')[:-1])
        raw_nirx = mne.io.read_raw_snirf(snirf_files[0])
        
        # Grab events
        event_files = glob(f"{flanker_dir}/*.evt")
        events = np.empty(1)
        for event_file in event_files:
            if event_file[-8:] == "_old.evt":
                continue
            events = pd.read_csv(event_files[0], sep = '\t', header = None, names = ['sample', '0', '1', '2', '3', '4', '5', '6', '7', '8'])
    
        if events.empty == False:
            raw_scans.append(raw_nirx)
            preproc_scans.append(preprocess(raw_nirx))
            scan_events.append(events)
        else:
            print(f"Skipping {subject_dir}")
        
    return subject_ids, raw_scans, preproc_scans, scan_events

def preprocess(scan):

    #try:
    # convert to optical density
    scan.load_data() 
    data, times = scan.get_data(return_times=True)
    plt.plot(times, data[0])
    plt.savefig('/storage1/fs1/perlmansusan/Active/moochie/github/hrc/tests/plots/raw_data.jpeg')
    plt.close()

    raw_od = mne.preprocessing.nirs.optical_density(scan)

    # scalp coupling index
    sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)
    raw_od.info['bads'] = list(compress(raw_od.ch_names, sci < 0.5))

    if len(raw_od.info['bads']) > 0:
        print("Bad channels in subject", raw_od.info['subject_info']['his_id'], ":", raw_od.info['bads'])

    # temporal derivative distribution repair (motion attempt)
    tddr_od = mne.preprocessing.nirs.tddr(raw_od)

    bp_od = tddr_od.filter(0.01, 0.5)

    # haemoglobin conversion using Beer Lambert Law (this will change channel names from frequency to hemo or deoxy hemo labelling)
    haemo = mne.preprocessing.nirs.beer_lambert_law(bp_od, ppf=0.1)

    # bandpass filter <-- seems to maj
    haemo_bp = haemo.copy().filter(
        0.05, 0.2, h_trans_bandwidth=0.1, l_trans_bandwidth=0.02)

    plt.plot(times, scipy.ndimage.gaussian_filter1d(haemo.get_data()[0], sigma=10))
    plt.savefig('/storage1/fs1/perlmansusan/Active/moochie/github/hrc/tests/plots/preproc_data.jpeg')
    plt.close()

    return haemo_bp

def test():
    subject_ids, raw_scans, preproc_scans, scan_events = load('/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R56/NIRS_data/')

    lens = observer.lens()
    montage = hrc.montage(preproc_scans[0])

    HRF = hrf.HRF()

    for subject_id, raw_nirx, preproc_nirx, nirx_events in zip(subject_ids, raw_scans, preproc_scans, scan_events):

        # Load subject
        deconvolved_nirx = preproc_nirx.copy()
        deconvolved_nirx.load_data()

        # Prep events
        temp_events = []
        for ind, value in nirx_events.iterrows():
            temp_events.append(int(value['sample']))
        events = [1 if ind in temp_events else 0 for ind in range(temp_events[-1])]
        events += [0 for _ in range(deconvolved_nirx.n_times - len(events))]

        # Convolve the scan
        montage.deconvolve_hrf(deconvolved_nirx, events)
        
        print(f"{subject_id} - {raw_nirx} - {preproc_nirx} - {deconvolved_nirx}")

        lens.compare_subject(subject_id, raw_nirx, preproc_nirx, deconvolved_nirx)

    montage.generate_distribution()

    lens.compare_subjects()

if __name__ == '__main__':
    test()


