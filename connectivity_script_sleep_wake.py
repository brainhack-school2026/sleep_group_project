import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from nilearn import image, plotting
from nilearn.maskers import NiftiMapsMasker, NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn import datasets, clean
from nilearn.interfaces.fmriprep import load_confounds_strategy
from itertools import groupby

from pathlib import Path
# Set this to path of where dataset is stored


dataset_root = Path("/Users/graceburns/Desktop/brainhack/ds003768-download/sourcedata/fMRI_processed_file_1/")

## Set a writable directory for the cache
writable_cache = '/tmp/nilearn_cache'

if not os.path.exists(writable_cache):
    os.makedirs(writable_cache)

### Shaefer Atals
atlas_dataset = datasets.fetch_atlas_schaefer_2018(data_dir=dataset_root, n_rois=400, yeo_networks=7)
atlas_filepath = atlas_dataset.maps
atlas_labels = atlas_dataset.labels[1:]

atlas_masker = NiftiLabelsMasker(
    labels_img=atlas_dataset.maps,
    standardize=True, # Z-scores the time series
    detrend=True,
    #high_pass=0.009, low_pass=0.08, t_r=2.1 # Adjust TR to your data
)

atlas_masker.fit()

network_assignments = [label.decode('utf-8').split('_')[2] if isinstance(label, bytes) else label.split('_')[2] for label in atlas_labels]
sort_idx = np.argsort(network_assignments)

def compute_connectivity(func_file):
    data_in_atlas_wake = atlas_masker.fit_transform(func_file_wake)
        
    confounds_matrix, sample_mask = load_confounds_strategy(
        img_files=func_file_wake,
        strategy=["global_signal"] # Includes GSR in the confound cleaning
    )

    # Apply to your data
    cleaned_img = clean(
        signals=func_file_wake,
        confounds=confounds_matrix,
        detrend=True,
        standardize=True
    )

    correlation_measure_wake = ConnectivityMeasure(kind='correlation', standardize=False,)
    correlation_matrix_wake = correlation_measure_wake.fit_transform([data_in_atlas_wake])[0]

    return correlation_matrix_wake

def plot_connectivity_matrix(matrix, labels, title="Connectivity Matrix"):
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title(title)
    plt.tight_layout()
    #plt.show()

def plot_networks(matrix, ax):

    ordered_matrix = matrix[sort_idx, :][:, sort_idx]
    ordered_labels = [network_assignments[i] for i in sort_idx]

    #count and summarize labels for axis purposes
    unique_networks = []
    network_sizes = []
    for k, g in groupby(ordered_labels):
        unique_networks.append(k)
        network_sizes.append(sum(1 for _ in list(g)))

    #establish label boundaries 
    boundaries = np.insert(np.cumsum(network_sizes), 0, 0)
    tick_positions = [(boundaries[i] + boundaries[i+1]) / 2 for i in range(len(boundaries) - 1)]

    #plot matrix
    display = plotting.plot_matrix(
        ordered_matrix,
        colorbar=True,
        reorder=False,
        axes=ax,
        vmin=-0.8, vmax=0.8,
        cmap="RdBu_r",

    )

    # boundary lines inside axis
    colors = plt.cm.Set2(np.linspace(0, 1, len(network_sizes)))
    for i in range(1, len(boundaries) - 1):
        ax.axvline(boundaries[i] - 0.5, color="black", linewidth=0.5, alpha=0.9)
        ax.axhline(boundaries[i] - 0.5, color="black", linewidth=0.5, alpha=0.9)
        
    bar_width = 15 
    matrix_size = ordered_matrix.shape[0]

    # colored bars along the edges
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i+1]
        current_color = colors[i]
        
        # Left edge color bar (Y-axis tracking)
        ax.axhspan(start - 0.5, end - 0.5, xmin=-bar_width/matrix_size, xmax=0, 
                color=current_color, clip_on=False, zorder=3)
        
        # Bottom edge color bar (X-axis tracking)
        ax.axvspan(start - 0.5, end - 0.5, ymin=-bar_width/matrix_size, ymax=0, 
                color=current_color, clip_on=False, zorder=3)

    ax.tick_params(axis='both', which='both', length=18, pad=10, direction='out')

    # labels 
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(unique_networks, rotation=90, fontsize=10)

    ax.set_yticks(tick_positions)
    ax.set_yticklabels(unique_networks, fontsize=10)


# atlas_dataset = datasets.fetch_atlas_msdl(data_dir=writable_cache)
# atlas_maps = image.load_img(atlas_dataset.maps)
# atlas_labels = atlas_dataset.labels

# atlas_masker = NiftiMapsMasker(
#     maps_img=atlas_maps,
#     standardize=True,
#     detrend=True,
#     memory=writable_cache,
#     memory_level=1
# )



correlation_measure = ConnectivityMeasure(kind="correlation")

subjects = sorted(dataset_root.glob("sub-*"))

wake_files = pd.read_csv("wake_stage_identification.csv", delimiter=",", index_col='subject')
sleep_files = pd.read_csv("sleep_stage_identification.csv", delimiter=",", index_col='subject')
fig, axes = plt.subplots(32, 3, figsize=(15, 110), layout="tight")
i = 0

wake_matrices = []
sleep_matrices = []

for subject in subjects:
    axes[i, 0].set_title(f"{subject.name} - Wake")
    axes[i, 1].set_title(f"{subject.name} - Sleep")
    axes[i, 2].set_title(f"{subject.name} - Difference")
    print("Found subject:", subject.name)
    
    try:

        selected_file = wake_files['session'].loc[subject.name]
        func_file_wake = dataset_root / f"{subject.name}/{subject.name}_{selected_file}_processed.nii.gz"
        print("Using:", func_file_wake)

        correlation_matrix_wake = compute_connectivity(func_file_wake)
        plot_networks(correlation_matrix_wake, axes[i, 0])

        wake_matrices.append(correlation_matrix_wake)
        
    except (KeyError, ValueError):
        print(f"Wake file not found for {subject.name}, skipping wake condition.")
        axes[i, 0].set_title("Wake data not available")
    
    try: 
        selected_file = sleep_files['session'].loc[subject.name]
        func_file_sleep = dataset_root / f"{subject.name}/{subject.name}_{selected_file}_processed.nii.gz"
        print("Using:", func_file_sleep)

        correlation_matrix_sleep = compute_connectivity(func_file_sleep)
        plot_networks(correlation_matrix_sleep, axes[i, 1])
        
        sleep_matrices.append(correlation_matrix_sleep)
        correlation_matrix_diff = correlation_matrix_wake - correlation_matrix_sleep
        plot_networks(correlation_matrix_diff, axes[i, 2])
        
    except (KeyError,ValueError):
        print(f"Sleep file not found for {subject.name}, skipping sleep condition.")
        axes[i, 1].set_title("Sleep data not available")
    i += 1
 
average_wake_matrix = np.mean(wake_matrices, axis=0)
average_sleep_matrix = np.mean(sleep_matrices, axis=0)
average_diff_matrix = average_wake_matrix - average_sleep_matrix

plt.savefig("connectivity_matrices_classifier_global_regression.pdf", dpi=300)

fig, ax = plt.subplots(1,3, figsize=(12, 10))
plot_networks(average_wake_matrix, ax[0])
plot_networks(average_sleep_matrix, ax[1])
plot_networks(average_diff_matrix, ax[2])
ax[0].set_title("Average Wake COnnectivity")
ax[1].set_title("Average Sleep Connectivity")
ax[2].set_title("Average Wake - Sleep Connectivity Difference")

plt.savefig("average_connectivity_difference_global_regression.pdf", dpi=300)
