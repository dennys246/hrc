import mne, sys
from glob import glob
from itertools import compress

sys.path.append("/storage1/fs1/perlmansusan/Active/moochie/github/hrc/hrconv")

import hrc, observer

def load(bids_dir, ex_subs = []):
    # make a list where all of the scans will get loaded into
    subject_ids = []
    raw_scans = []
    preproc_scans = []

    # Load in master file with scan order info
    subject_dirs = glob(f'{bids_dir}*/')[:3]
    print(subject_dirs)

    for dir_ind, directory in enumerate(subject_dirs[:2]):
        for excluded in ex_subs:
            if excluded in directory:
                print(f"Deleting {directory}")
                del subject_dirs[dir_ind]

    for subject_dir in subject_dirs:

        subject_ids.append(subject_dir.split('/')[-2])

        mat_files = glob(subject_dir + '*/*/*_probeInfo.mat') + glob(subject_dir + '*/*/*_probeinfo.mat')
        if len(mat_files) == 0:
            print(f"Missing probe info for {subject_dir}...\n")
            continue
        
        print(subject_dir)
        subject_dir = '/'.join(mat_files[0].split('/')[:-1])

        raw_nirx = mne.io.read_raw_nirx(subject_dir)
        raw_scans.append(raw_nirx)

        preproc_scans.append(preprocess(raw_nirx))

    return subject_ids, raw_scans, preproc_scans

def preprocess(scan):

    #try:
    # convert to optical density
    scan.load_data() 

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

    # bandpass filter
    haemo_bp = haemo.copy().filter(
        0.05, 0.7, h_trans_bandwidth=0.2, l_trans_bandwidth=0.02)

    return haemo_bp
    #except:
    #    print(f"Scan failed to preprocess...\n{scan}")

def test():
    subject_ids, raw_scans, preproc_scans = load('/storage1/fs1/perlmansusan/Active/moochie/analysis/CARE/NIRS_data_clean_2/')

    lens = observer.lens()

    for subject_id, raw_nirx, preproc_nirx in zip(subject_ids, raw_scans, preproc_scans):
        # Create a copy of the original scan
        deconvolved_nirx = preproc_nirx.copy()
        deconvolved_nirx.load_data()

        # Convolve the scan
        deconvolved_nirx = hrc.deconvolve_hrf(deconvolved_nirx, plot = True)
        
        print(f"{subject_id} - {raw_nirx} - {preproc_nirx} - {deconvolved_nirx}")

        lens.compare_subject(subject_id, raw_nirx, preproc_nirx, deconvolved_nirx)

    lens.compare_subjects()

if __name__ == '__main__':
    test()


