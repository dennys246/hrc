import mne, sys, os, random
from glob import glob

sys.path.append("/storage1/fs1/perlmansusan/Active/moochie/github/hrc/hrconv")

import observer
import hrfunc as hrf

sys.path.append("/storage1/fs1/perlmansusan/Active/moochie/github/hrc/tests")

import pipeline

def deconvolve_care(hrf_filename, _overwrite = False):

    working_dir = "/storage1/fs1/perlmansusan/Active/moochie/github/hrc/tests/"

    subject_ids, child_scans, parent_scans = pipeline.load_care('/storage1/fs1/perlmansusan/Active/moochie/analysis/CARE/NIRS_data_clean_2/')

    # shuffle data to allow for mutliple deconvolution streams
    zipper = list(zip(subject_ids, child_scans, parent_scans))
    random.shuffle(zipper)
    subject_ids, child_scans, parent_scans = zip(*zipper)

    lens = observer.lens(working_dir)
    montage = hrf.montage(child_scans[0], hrf_filename)

    for subject_id, child_nirx, parent_nirx in zip(subject_ids, child_scans, parent_scans):
        for cp_ind, preproc_nirx in enumerate([child_nirx, parent_nirx]):
            # Construct deconvolved filename
            if cp_ind == 0:
                sid = subject_id
            else:
                sid = subject_id[:-1]
            deconvolved_filename = f"/storage1/fs1/perlmansusan/Active/moochie/analysis/CARE/NIRS_data_clean_2/{sid}/V0/{sid}_V0_fNIRS/{sid}_deconvolved.fif"
            
            if os.path.exists(deconvolved_filename) and _overwrite == False:
                print(f"Skipping {sid}, already deconvolved...")
                continue

            print(f"Deconvolving {sid}...")

            # Load subject
            deconvolved_nirx = preproc_nirx.copy()
            deconvolved_nirx.load_data()

            # Deconvolve the scan
            print(f"Deconlving subject {subject_id}...")
            deconvolved_nirx = montage.deconvolve_nirs(deconvolved_nirx, hrf_filename)
            
            print(f"{subject_id} - {preproc_nirx} - {deconvolved_nirx}")

            #lens.compare_subject(subject_id, raw_nirx, preproc_nirx, deconvolved_nirx, events, length = 1500)
            print(f"Deconvolved filename: {deconvolved_filename}")
            deconvolved_nirx.save(deconvolved_filename, overwrite = _overwrite)

    #lens.compare_subjects()

if __name__ == '__main__':
    hrf_filename = "/storage1/fs1/perlmansusan/Active/moochie/github/hrc/tests/P-CAT_hrfs.json"
    deconvolve_care(hrf_filename)


