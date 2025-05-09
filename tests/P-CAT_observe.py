import mne, sys, os
from glob import glob

sys.path.append("/storage1/fs1/perlmansusan/Active/moochie/github/hrc/hrconv")

import observer, pipeline
import hrfunc as hrf


def compare_subjects():
    subject_ids, raw_scans, preproc_scans, scan_events = pipeline.load_pcat('/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R56/NIRS_data/')

    lens = observer.lens()
    montage = hrf.montage(preproc_scans[0], "P-CAT_hrfs.json")

    for subject_id, raw_nirx, preproc_nirx, nirx_events in zip(subject_ids, raw_scans, preproc_scans, scan_events):
        # Construct deconvolved filename
        deconvolved_filename = f"/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R56/NIRS_data/{subject_id}/{subject_id}_Flanker/{subject_id}_Flanker_Deconvolved.fif"
        
        if os.path.exists(deconvolved_filename) == False:
            continue

        # Load subject
        deconvolved_nirx = mne.io.read_raw_fif(deconvolved_filename)
        deconvolved_nirx.load_data()

        # Prep events
        temp_events = []
        for ind, value in nirx_events.iterrows():
            temp_events.append(int(value['sample']))
        events = [1 if ind in temp_events else 0 for ind in range(temp_events[-1])]
        events += [0 for _ in range(deconvolved_nirx.n_times - len(events))]
        
        print(f"{subject_id} - {raw_nirx} - {preproc_nirx} - {deconvolved_nirx}")

        lens.compare_subject(subject_id, raw_nirx, preproc_nirx, deconvolved_nirx, events, length = 1500)

    print("Comparing all subjects")

    lens.compare_subjects()

if __name__ == '__main__':
    compare_subjects()