import mne, sys, os, math, pipeline, nilearn
from glob import glob
import numpy as np
import pandas as pd
from mne_nirs.statistics import run_glm
import matplotlib.pyplot as plt

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

    # Grab the subject non-deconv scan
    preproc_scan = preproc_scans[ind]

    # Grab subject events for just congruent vs. incongruent
    events, event_id = pipeline.process_congruency(scan_events[ind])

    # Format events for design matrix
    pandas_events = pd.DataFrame([[event[0] * sfreq, event_duration, event[2], 1] for event in events], columns = ['onset', 'duration', 'trial_type', 'modulation'])
    frame_times = np.array([sample / sfreq for sample in range(preproc_scan.n_times)])
    print(pandas_events)

    # Create design matrix
    design_matrix = nilearn.glm.first_level.make_first_level_design_matrix(
        frame_times,
        pandas_events, 
        hrf_model = 'spm', 
        drift_model='cosine', 
        high_pass=0.01, 
        drift_order=1)

    # Calculcate subject congruency-incongruency contrast
    glm_results = run_glm(preproc_scan, design_matrix)

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
    valid_idxs = get_channels_with_positions(preproc_scan.info) 
    subject_contrast[valid_idxs] = individual_contrasts  # fill in where you have data
    subject_contrasts.append(subject_contrast)

print("Calculating subject pool wise contrast and plotting...")
# Calculate mean and standard deviation of contrast across subjects
contrast_array = np.vstack(subject_contrasts)  # shape (n_subjects, n_channels)
mean_contrast = np.mean(contrast_array, axis=0)  # shape (n_channels,)
std_contrast = np.std(contrast_array, axis=0)

# Pick just one type of channel, i.e. hbo
picks = mne.pick_types(preproc_scan.info, fnirs='hbo')

# Use only the hbo channels for contrast and info
contrast_for_plot = mean_contrast[picks]
info_for_plot = mne.pick_info(preproc_scan.info, sel=picks)

# Calculate max value in contrast for plotting
max_abs = np.abs(np.max(mean_contrast))

# Plot a topo map of contrasts
image, cn = mne.viz.plot_topomap(contrast_for_plot, info_for_plot, show=True, cmap='RdBu_r', vlim=(-max_abs, max_abs))
plt.title(f"P-CAT Standard Contrast")

# Save and close plot
image.figure.savefig(f"{plot_dir}P-CAT_standard_congruent-incongruent_contrast_topomap.jpg", dpi=300, bbox_inches='tight')
plt.close(image.figure)

print("Contrast values for plotting:", contrast_for_plot)
print("Number of valid channels:", np.sum(~np.isnan(contrast_for_plot)))
print("Info object contains channels:", [ch['ch_name'] for ch in info_for_plot['chs']])

for i, ch in enumerate(info_for_plot['chs']):
    print(f"{ch['ch_name']} position: {ch['loc'][:3]}")