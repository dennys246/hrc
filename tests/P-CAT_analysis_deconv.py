import mne, os, pipeline, nilearn
from glob import glob
import numpy as np
import pandas as pd
import nibabel as nib

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pyvista as pv

from glob import glob
from nibabel.affines import apply_affine
from nilearn import plotting, datasets, surface
from mne_nirs.statistics import run_glm


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

excluded = ['1189', '1126']

# Load in our data
subject_ids, raw_scans, preproc_scans, scan_events = pipeline.load_pcat(data_dir)

# Create hold variable for subject level contrasts
print(preproc_scans)
standard_contrasts = []
deconv_contrasts = []
for ind, subject_id in enumerate(subject_ids):
    if str(subject_id) in excluded:
        print(f"Subject {subject_id} excluded, skipping...")
        continue

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
    print(f"Events...\n{events}")

    # Format events for design matrix
    pandas_events = pd.DataFrame([[event[0], event_duration, event[2], 1] for event in events], columns = ['onset', 'duration', 'trial_type', 'modulation'])
    frame_times = np.array([sample / sfreq for sample in range(deconv_scan.n_times)])
    print(pandas_events)

    # Create design matrix
    deconv_design_matrix = nilearn.glm.first_level.make_first_level_design_matrix(
        frame_times,
        pandas_events, 
        hrf_model = None, 
        drift_model = 'cosine', 
        high_pass = 0.01, 
        drift_order = 1)
    
    standard_design_matrix = nilearn.glm.first_level.make_first_level_design_matrix(
        frame_times,
        pandas_events, 
        hrf_model = "spm", 
        #hrf_model = "spm + derivative + dispersion", 
        drift_model = 'cosine', 
        high_pass = 0.01, 
        drift_order = 1)

    for scan, design_matrix, contrast_store, preprocessing in zip([deconv_scan, preproc_scan], [deconv_design_matrix, standard_design_matrix], [deconv_contrasts, standard_contrasts], ['Deconvolved', 'Standard']):
        # Calculcate subject congruency-incongruency contrast
        glm_results = run_glm(scan, design_matrix)

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
        
        print(f"Subject {subject_id}: number of channel contrasts = {len(individual_contrasts)}")
        print(f"Expected number of valid channels = {len(valid_idxs)}")
        print(f"Valid channel indices for {subject_id}: {valid_idxs}")

        contrast_store.append(subject_contrast)


deconv_contrasts = np.vstack(deconv_contrasts)
deconv_mean = np.mean(deconv_contrasts, axis=0)
deconv_mean *= 100
deconv_abs = max(abs(np.min(deconv_contrasts)), abs(np.max(deconv_contrasts)))
deconv_vlim = (-deconv_abs, deconv_abs)

# Format contrast from standard preprocessed data contrasts
standard_contrasts = np.vstack(standard_contrasts)
standard_mean = np.mean(standard_contrasts, axis=0)
standard_mean *= 100
standard_abs = max(abs(np.min(standard_contrasts)), abs(np.max(standard_contrasts)))
standard_vlim = (-standard_abs, standard_abs)
    
# Calculate max value in contrast for plotting
combined = np.concatenate([standard_mean, deconv_mean])
shared_min = np.min(combined)
shared_max = np.max(combined)
shared_abs = max(abs(shared_min), abs(shared_max))
_vlim = (-shared_abs, shared_abs)

# Create info for plot
#picks = mne.pick_types(preproc_scans[2].info, fnirs='hbo')
picks = [0, 2, 4,  6,  8, 10, 12, 14, 16, 18]
info_for_plot = mne.pick_info(preproc_scans[2].info, sel=picks)
#for ch in info_for_plot['chs']:
#    ch['loc'][1] -= 0.02  # shift back by 1 unit on the y-axis
print(picks)

#Generate plot
fig, _axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot a topo map of contrasts using mask
deconv_image, deconv_cn = mne.viz.plot_topomap(
    deconv_mean[picks],
    info_for_plot,
    axes = _axes[0, 0],
    show = False,
    cmap = 'RdBu_r',
    vlim = deconv_vlim,
    contours = 0,
    extrapolate = 'local'
)
_axes[0, 0].set_title(f"Deconvolved GLM Contrast")

standard_image, standard_cn = mne.viz.plot_topomap(
    standard_mean[picks],
    info_for_plot,
    axes = _axes[0, 1],
    show = False,
    cmap = 'RdBu_r',
    vlim = standard_vlim,
    contours = 0,
    extrapolate = 'local'
)
_axes[0, 1].set_title(f"Standard GLM Contrast with Glover HRF")

norm = plt.Normalize(vmin = -shared_abs, vmax = shared_abs)

for i, ch in enumerate(info_for_plot['chs']):
    print(f"{ch['ch_name']} position: {ch['loc'][:3]}")

# Get positions of channels
positions = np.array([ch['loc'][:2] for ch in info_for_plot['chs']])  # x, y
colors = cm.RdBu_r(norm(deconv_contrasts[0][picks]))[:, :3]  # RGB, no alpha

deconv_sc = _axes[1, 0].scatter(positions[:, 0], 
                positions[:, 1],
                c = deconv_mean[picks], 
                cmap = 'RdBu_r',
                vmin = -deconv_abs, 
                vmax = deconv_abs,
                s = 100, 
                edgecolors = 'k')
_axes[1, 0].axis('equal')
_axes[1, 0].grid(True)

standard_sc = _axes[1, 1].scatter(positions[:, 0], 
                positions[:, 1],
                c = standard_mean[picks], 
                cmap = 'RdBu_r',
                vmin = -standard_abs, 
                vmax = standard_abs,
                s = 100, 
                edgecolors = 'k')
_axes[1, 1].axis('equal')
_axes[1, 1].grid(True)

norm = plt.Normalize(vmin=-shared_abs, vmax=shared_abs)
sm = cm.ScalarMappable(cmap='RdBu_r', norm=norm)
sm.set_array([])  # Needed for matplotlib < 3.1

deconv_norm = plt.Normalize(vmin=-deconv_abs, vmax=deconv_abs)
deconv_sm = cm.ScalarMappable(cmap='RdBu_r', norm=norm)
deconv_sm.set_array([])  # Needed for matplotlib < 3.1

standard_norm = plt.Normalize(vmin=-standard_abs, vmax=standard_abs)
standard_sm = cm.ScalarMappable(cmap='RdBu_r', norm=norm)
standard_sm.set_array([])  # Needed for matplotlib < 3.1


# Add a colorbar to the figure
deconv_cbar = fig.colorbar(deconv_sm, 
                ax = _axes[1, 0], 
                orientation = 'vertical', 
                shrink = 0.6, 
                label = 'Contrast Value')

standard_cbar = fig.colorbar(standard_sm, 
                ax = _axes[1, 1], 
                orientation = 'vertical', 
                shrink = 0.6, 
                label = 'Contrast Value')


"""

# Create all metadata for MNI projects
montage = mne.channels.read_dig_fif(deconvolved_filename)
trans = mne.channels.compute_native_head_t(montage)
mni_coordinates = mne.head_to_mni(np.array(channel['loc'][:3] for channel in info_for_plot['chs']), 'fsaverage', trans, '/storage1/fs1/perlmansusan/Active/moochie/github/hrc/tests/fsaverage')
shape = (91, 109, 91)
affine = np.array([
    [ 2,  0,  0, -90],
    [ 0,  2,  0, -126],
    [ 0,  0,  2, -72],
    [ 0,  0,  0,   1]
])
vox_coords = np.round(apply_affine(np.linalg.inv(affine), mni_coordinates)).astype(int)

# Translate deconv contrasts to MNI space
deconv_data = np.zeros(shape)
for (x, y, z), val in zip(vox_coords, deconv_contrasts):
    if (0 <= x < shape[0]) and (0 <= y < shape[1]) and (0 <= z < shape[2]):
        deconv_data[x, y, z] = val

deconv_image = nib.Nifti1Image(deconv_data, affine) # Build image
deconv_header = deconv_image.header
nib.save(deconv_image, f'{plot_dir}fnirs_deconv_contrast_map.nii.gz')

# Translate standard contrasts to MNI space
standard_data = np.zeros(shape)
for (x, y, z), val in zip(vox_coords, standard_contrasts):
    if (0 <= x < shape[0]) and (0 <= y < shape[1]) and (0 <= z < shape[2]):
        standard_data[x, y, z] = val

standard_image = nib.Nifti1Image(standard_data, affine) # Build image
standard_header = standard_image.header
nib.save(standard_image, f'{plot_dir}fnirs_standard_contrast_map.nii.gz')

_threshold = 0
intensity = 0.5

fsaverage = datasets.fetch_surf_fsaverage()

fig, _axes = plt.subplots(2, 1, figsize=(10, 8))

deconv_texture = surface.vol_to_surf(deconv_image, fsaverage.pial_left)
plotting.plot_surf_stat_map(surf_map = fsaverage.infl_left,
                            stat_map = deconv_texture, 
                            hemi = 'left', 
                            view = 'anterior', 
                            title = "Deconvolved GLM Contrasts", 
                            colorbar = False, 
                            threshold = _threshold, 
                            bg_on_data = True, 
                            cmap='Spectral', 
                            axes = _axes[0],
                            vmax = shared_abs, 
                            vmin = -shared_abs)
_axes[0].set_title(f"Deconvolved GLM Contrast")

standard_texture = surface.vol_to_surf(standard_image, fsaverage.pial_left)
plotting.plot_surf_stat_map(surf_map = fsaverage.infl_left, 
                            stat_map = standard_texture, 
                            hemi = 'left', 
                            view = 'anterior', 
                            title = "Standard GLM Contrasts", 
                            colorbar = True, 
                            threshold = _threshold, 
                            bg_on_data = True, 
                            cmap='Spectral',
                            axes = _axes[1],
                            vmax = shared_abs, 
                            vmin = -shared_abs)
_axes[1].set_title(f"Standard GLM Contrast with Glover HRF")
"""

# Add overall title
fig.suptitle("P-CAT Flanker Congruent-Incongruent Contrast", fontsize = 14)

# Add shared colorbar for scatter plots
_axes[1, 1].set_xlabel('Contrast (% signal change)')

# Layout and save
plt.tight_layout()
plt.savefig(f"{plot_dir}P-CAT_combined_congruent-incongruent_contrasts.jpg")
plt.close(fig)

"""

plotter = mne.viz.plot_alignment(
    info = info_for_plot,
    surfaces = [],  # or add 'head' if you want a background
    coord_frame = 'head',
    show_axes = True,
)

# Add colored dots at channel locations
plotter.plotter.add_points(
    points=positions,
    scalars=contrast_for_plot,
    cmap = 'RdBu_r',
    point_size = 15,
    render_points_as_spheres = True
)


plotter.plotter.add_scalar_bar(title = "Contrast")
plotter.plotter.screenshot(f"{plot_dir}P-CAT__{preprocessing.lower()}_contrast_colored_dots_3D.jpg")
plotter.plotter.close() 

"""