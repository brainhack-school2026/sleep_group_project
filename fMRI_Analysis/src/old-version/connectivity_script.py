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
from nilearn import datasets

from pathlib import Path
# Set this to path of where dataset is stored

def compute_connectivity(func_file):
    fmri_img = image.load_img(func_file)
    timeseries = atlas_masker.transform(fmri_img)
    matrix = correlation_measure.fit_transform([timeseries])[0]
    return timeseries, matrix

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


dataset_root = Path("/Users/graceburns/Desktop/brainhack/ds003768-download/sourcedata/fMRI_processed_file_1/")

# Set a writable directory for the cache
writable_cache = '/tmp/nilearn_cache'

if not os.path.exists(writable_cache):
    os.makedirs(writable_cache)

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

### Shaefer Atals
atlas_dataset = datasets.fetch_atlas_schaefer_2018(data_dir=writable_cache, n_rois=400, yeo_networks=7)
atlas_filepath = atlas_dataset.maps
atlas_labels = atlas_dataset.labels[1:]

atlas_masker = NiftiLabelsMasker(
    labels_img=atlas_dataset.maps,
    standardize=True, # Z-scores the time series
    detrend=True,
    #high_pass=0.009, low_pass=0.08, t_r=2.1 # Adjust TR to your data
)

atlas_masker.fit()

correlation_measure = ConnectivityMeasure(kind="correlation")

subjects = sorted(dataset_root.glob("sub-*"))

fig, axes = plt.subplots(32, 3, figsize=(15, 110), layout="tight")
i = 0
for subject in subjects:
    
    axes[i, 0].set_title(f"{subject.name} - Wake")
    axes[i, 1].set_title(f"{subject.name} - Sleep")
    axes[i, 2].set_title(f"{subject.name} - Difference")
    print("Found subject:", subject.name)
    func_file_wake = dataset_root / f"{subject.name}/{subject.name}_task-rest_run-1_processed.nii.gz"

    print("Using:", func_file_wake)

    #timeseries, matrix = compute_connectivity(func_file)

    data_in_atlas_wake = atlas_masker.fit_transform(func_file_wake)

    correlation_measure_wake = ConnectivityMeasure(kind='correlation', standardize=False,)
    correlation_matrix_wake = correlation_measure_wake.fit_transform([data_in_atlas_wake])[0]
    plotting.plot_matrix(correlation_matrix_wake, labels=None,
                     vmax=1, vmin=-1, axes=axes[i, 0])
    try: 
        func_file_sleep = dataset_root / f"{subject.name}/{subject.name}_task-sleep_run-1_processed.nii.gz"

        print("Using:", func_file_sleep)

        #timeseries, matrix = compute_connectivity(func_file)

        data_in_atlas_sleep = atlas_masker.fit_transform(func_file_sleep)

        correlation_measure_sleep = ConnectivityMeasure(kind='correlation', standardize=False,)
        correlation_matrix_sleep = correlation_measure_sleep.fit_transform([data_in_atlas_sleep])[0]
        plotting.plot_matrix(correlation_matrix_sleep, labels=None,
                        vmax=1, vmin=-1, axes=axes[i, 1])
        
        correlation_matrix_diff = correlation_matrix_wake - correlation_matrix_sleep
        plotting.plot_matrix(correlation_matrix_diff, labels=None,
                        vmax=1, vmin=-1, axes=axes[i, 2])
        
    except ValueError:
        print(f"Sleep file not found for {subject.name}, skipping sleep condition.")
        axes[i, 1].set_title("Sleep data not available")
    i += 1
 
plt.savefig("connectivity_matrices_shaefer_not_standardized.pdf", dpi=300)


