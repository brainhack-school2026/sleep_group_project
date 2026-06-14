import mne
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mne_connectivity import spectral_connectivity_epochs

def compute_coherence_matrices(epochs_path, frequency_bands):
    """
    Loads epoched data and computes a coherence matrix for specified frequency bands.
    """
    print(f"Loading epochs from: {epochs_path.name}")
    epochs = mne.read_epochs(epochs_path, preload=True)
    
    # Pick only EEG channels to avoid computing connectivity for EOG/Stim channels
    epochs_eeg = epochs.copy().pick("eeg")
    ch_names = epochs_eeg.ch_names
    sfreq = epochs_eeg.info["sfreq"]
    
    # Dictionary to hold the final matrix for each band
    connectivity_matrices = {}
    
    # Compute connectivity for each frequency band separately
    for band_name, (fmin, fmax) in frequency_bands.items():
        print(f"  Computing coherence for {band_name} band ({fmin}-{fmax} Hz)...")
        
        # 'coh' = Standard Coherence. 
        # Alternatives: 'imcoh' (Imaginary Coherence, great for avoiding volume conduction)
        #               'pli' (Phase Lag Index)
        con = spectral_connectivity_epochs(
            epochs_eeg,
            method="coh", 
            mode="multitaper",  # Multitaper is robust and highly recommended for sleep/rest
            sfreq=sfreq,
            fmin=fmin,
            fmax=fmax,
            faverage=True,      # Average across the frequencies within the band
            verbose=False
        )
        
        # con.get_data() returns an array of shape (n_signals * (n_signals - 1) / 2,) 
        # because it only computes the unique pairs by default.
        # .get_data('dense') expands this into a full symmetric matrix of shape (n_channels, n_channels, n_freqs)
        matrix_3d = con.get_data(output="dense")
        
        # Squeeze out the trailing frequency dimension since faverage=True
        matrix_2d = np.squeeze(matrix_3d)
        
        connectivity_matrices[band_name] = matrix_2d
        
    return connectivity_matrices, ch_names

def plot_connectivity_matrix(matrix, ch_names, title, save_path=None):
    """
    Plots and optionally saves a beautiful heatmap of the connectivity matrix.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="RdBu_r")
    
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(np.arange(len(ch_names)))
    ax.set_yticks(np.arange(len(ch_names)))
    ax.set_xticklabels(ch_names, rotation=90, fontsize=8)
    ax.set_yticklabels(ch_names, fontsize=8)
    
    fig.colorbar(im, ax=ax, label="Coherence Value")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def main():
    # Define standard frequency bands relevant to Sleep and Rest
    # Adjusted boundaries to map cleanly onto sleep stages
    frequency_bands = {
        "Delta": (0.5, 4.0),
        "Theta": (4.0, 8.0),
        "Alpha": (8.0, 12.0),
        "Beta": (12.0, 30.0)
    }
    
    # Path setup
    data_dir = Path("C:/Users/...") # update with your path here 
    output_dir = data_dir / "connectivity_results"
    output_dir.mkdir(exist_ok=True) # Create output directory if it doesn't exist

    subjects_file = data_dir / "WAKEsubjects.txt"
    if not subjects_file.exists():
        print(f"Error: {subjects_file} not found.")
        return

    with open(subjects_file, "r") as f:
        subjects = [line.strip() for line in f if line.strip()]

    for subject in subjects:
        # Match the output naming convention from your preprocessing script
        epochs_path = data_dir / f"{subject}_preprocessed_epo.fif"
        
        if not epochs_path.exists():
            print(f"Preprocessed file missing for {subject}. Skipping...")
            continue
            
        try:
            matrices, ch_names = compute_coherence_matrices(epochs_path, frequency_bands)
            
            # Save matrices and figures for each band
            for band_name, matrix in matrices.items():
                # 1. Save numerical data as a NumPy binary file (.npy) for later statistical modeling
                npy_filename = output_dir / f"{subject}_{band_name}_coherence.npy"
                np.save(npy_filename, matrix)
                
                # 2. Generate and save the matrix plot image
                img_filename = output_dir / f"{subject}_{band_name}_coherence.png"
                plot_title = f"{subject} - {band_name} Coherence Matrix"
                plot_connectivity_matrix(matrix, ch_names, title=plot_title, save_path=img_filename)
                
            # Optional: Save channel labels order once so you know the matrix indexing layout
            if not (output_dir / "channel_order.txt").exists():
                with open(output_dir / "channel_order.txt", "w") as f:
                    f.write("\n".join(ch_names))

            print(f"Successfully generated and saved connectivity for {subject}!\n" + "-"*40)
            
        except Exception as e:
            print(f"Failed to compute connectivity for {subject}: {e}")
            continue

if __name__ == "__main__":
    main()