import mne, os, pipeline
from glob import glob
import numpy as np
import pandas as pd
from mne_nirs.statistics import run_glm
from mne.evoked import EvokedArray
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import matplotlib.pyplot as plt
import mne
import numpy as np
from nilearn.plotting import plot_design_matrix

"""
This python script is used to run a generic GLM analysis on fNIRS
data collected apart of the P-CAT study to access the impact of
HRF deconvolution on analysis outcomes.
"""

def get_channels_with_positions(info):
    """Return indices of channels with valid position information."""
    return [idx for idx, ch in enumerate(info['chs']) if np.any(ch['loc'][:3])]

# Define runtime variables
data_dir = '/storage1/fs1/perlmansusan/Active/moochie/study_data/P-CAT/R56/NIRS_data/'
plot_dir = "/storage1/fs1/perlmansusan/Active/moochie/github/hrc/tests/plots/"
#data_dir = '/Volumes/perlmansusan/Active/moochie/study_data/P-CAT/R56/NIRS_data/'
#plot_dir = "/Volumes/perlmansusan/Active/moochie/github/hrc/tests/plots/"
sfreq = 7.81
event_duration = 3.0

# Load in our data
subject_ids, raw_scans, preproc_scans, scan_events = pipeline.load(data_dir)

# Create hold variable for subject level contrasts
print(preproc_scans)
subject_contrasts = []
for ind, subject_id in enumerate(subject_ids):
    print(f"Calculating subject {subject_id} congruent-incongruent contrasts")
    # Generate filename where deconvolved file is stored
    deconvolved_filename = f"{data_dir}{subject_id}/{subject_id}_Flanker/{subject_id}_Flanker_Deconvolved.fif"
    
    # Check if is exists
    if os.path.exists(deconvolved_filename) == False: 
        print(f"Missing deconvolved file for {subject_id}, skipping...")
        continue

    # Read in subjects deconv scan
    deconv_scan = mne.io.read_raw_fif(deconvolved_filename)

    # Grab the subject non-deconv scan
    preproc_scan = preproc_scans[ind]

    # Grab subject events for just congruent vs. incongruent
    events, event_id = pipeline.process_congruency(scan_events[ind])

    # Format events for design matrix
    pandas_events = pd.DataFrame([[event[0] * sfreq, event_duration, event[2], 1] for event in events], columns = ['onset', 'duration', 'trial_type', 'modulation'])
    frame_times = np.array([sample / sfreq for sample in range(deconv_scan.n_times)])
    print(pandas_events)

    # Create design matrix
    design_matrix = make_first_level_design_matrix(
        deconv_scan,
        drift_model="cosine",
        high_pass=0.005,  # Must be specified per experiment
        hrf_model="spm",
        stim_dur=5.0,
    )

    # Calculcate subject congruency-incongruency contrast
    glm_results = run_glm(deconv_scan, design_matrix)

    contrast_vec = np.array([1, -1] + [0] * (len(design_matrix.columns) - 2))

    individual_contrasts = []

    contrasts_obj = glm_results.compute_contrast(contrast_vec, 'F')
    contrasts = contrasts_obj.data
    contrast_effect = contrasts.effect
    contrast_variance = contrasts.variance

    print(f"Contrast effects: {contrast_effect}")
    for channel_contrast in contrast_effect:  # Each channel
        print(channel_contrast)
        individual_contrasts.append(channel_contrast)

    n_channels = len(raw_scans[ind].info['chs'])  # or 20
    subject_contrast = np.full(n_channels, np.nan)
    valid_idxs = get_channels_with_positions(deconv_scan.info) 
    subject_contrast[valid_idxs] = individual_contrasts  # fill in where you have data
    subject_contrasts.append(subject_contrast)

print("Calculating subject pool wise contrast and plotting...")
# Extract evoked data (EvokedArray) from each ContrastResults
evoked_list = [cr.data for cr in subject_contrasts]

# Check that all info objects are equal (i.e., channels, montage, etc.)
info = evoked_list[0].info
assert all(evk.info == info for evk in evoked_list)

# Stack and average the data
data_stack = np.stack([evk.data for evk in evoked_list], axis=0)  # shape: (n_subjects, n_channels, n_times)
avg_data = np.mean(data_stack, axis=0)

# Create a new EvokedArray with averaged data
evoked_avg = EvokedArray(avg_data, info, tmin=evoked_list[0].times[0])

# Plot
evoked_avg.plot_topo()
# Save and close plot
image.figure.savefig(f"{plot_dir}P-CAT_deconvolved_congruent-incongruent_contrast_topomap.jpg", dpi=300, bbox_inches='tight')
plt.close(image.figure)
