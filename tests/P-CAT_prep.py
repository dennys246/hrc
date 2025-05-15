import mne, sys, os, random
from glob import glob

sys.path.append("/storage1/fs1/perlmansusan/Active/moochie/github/hrc/hrconv")

import observer
import hrfunc as hrf

sys.path.append("/storage1/fs1/perlmansusan/Active/moochie/github/hrc/tests")

import pipeline

def estimate_hrf(hrf_filename, overwrite = False):
    
    if os.path.exists(hrf_filename) and overwrite == False: # Check if hrf
        return ValueError(f"HRF filename provided exists (if intentional set overwrite to True)...\nhrf_filename: {hrf_filename}")
    
    subject_ids, raw_scans, preproc_scans, scan_events = pipeline.load_pcat('/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R56/NIRS_data/', deconvolution = True)

    montage = hrf.montage(preproc_scans[0])

    for subject_id, raw_nirx, preproc_nirx, nirx_events in zip(subject_ids, raw_scans, preproc_scans, scan_events):
        preproc_nirx.load_data()

        # Prep events
        temp_events = []
        for ind, value in nirx_events.iterrows():
            temp_events.append(int(value['sample']))
        events = [1 if ind in temp_events else 0 for ind in range(temp_events[-1])]
        events += [0 for _ in range(preproc_nirx.n_times - len(events))]

        # Convolve the scan
        montage.deconvolve_hrf(preproc_nirx, events)
        
        print(f"{subject_id} - {raw_nirx} - {preproc_nirx} - {preproc_nirx}")

    montage.generate_distribution()

    montage.correlate_hrf("/storage1/fs1/perlmansusan/Active/moochie/github/hrc/tests/plots/montage_corr.png")
    montage.correlate_canonical("/storage1/fs1/perlmansusan/Active/moochie/github/hrc/tests/plots/canonical_corr.png")

    montage.save(hrf_filename)



def deconvolve_pcat(hrf_filename, _overwrite = False):

    working_dir = "/storage1/fs1/perlmansusan/Active/moochie/github/hrc/tests/"

    subject_ids, raw_scans, preproc_scans, scan_events = pipeline.load_pcat('/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R56/NIRS_data/', deconvolution = True)

    # shuffle data to allow for mutliple deconvolution streams
    zipper = list(zip(subject_ids, raw_scans, preproc_scans, scan_events))
    random.shuffle(zipper)
    subject_ids, raw_scans, preproc_scans, scan_events = zip(*zipper)

    lens = observer.lens(working_dir)
    montage = hrf.montage(preproc_scans[0], hrf_filename)

    for subject_id, raw_nirx, preproc_nirx, nirx_events in zip(subject_ids, raw_scans, preproc_scans, scan_events):
        # Construct deconvolved filename
        deconvolved_filename = f"/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R56/NIRS_data/{subject_id}/{subject_id}_Flanker/{subject_id}_Flanker_Deconvolved.fif"
        
        if os.path.exists(deconvolved_filename) and _overwrite == False:
            print(f"Skipping {subject_id}, already deconvolved...")
            continue

        print(f"Deconvolving {subject_id}...")

        # Load subject
        deconvolved_nirx = preproc_nirx.copy()
        deconvolved_nirx.load_data()

        # Prep events
        temp_events = []
        for ind, value in nirx_events.iterrows():
            temp_events.append(int(value['sample']))
        events = [1 if ind in temp_events else 0 for ind in range(temp_events[-1])]
        events += [0 for _ in range(deconvolved_nirx.n_times - len(events))]

        # Deconvolve the scan
        print(f"Deconlving subject {subject_id}...")
        deconvolved_nirx = montage.deconvolve_nirs(deconvolved_nirx, hrf_filename)
        
        print(f"{subject_id} - {raw_nirx} - {preproc_nirx} - {deconvolved_nirx}")

        lens.compare_subject(subject_id, raw_nirx, preproc_nirx, deconvolved_nirx, events, length = 1500)

        deconvolved_nirx.save(deconvolved_filename, overwrite = _overwrite)

    lens.compare_subjects()

if __name__ == '__main__':
    hrf_filename = "/storage1/fs1/perlmansusan/Active/moochie/github/hrc/tests/P-CAT_hrfs.json"
    #estimate_hrf(hrf_filename, True)
    deconvolve_pcat(hrf_filename)


