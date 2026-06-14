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
            block=True,
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
    Plots and optionally saves a heatmap of the connectivity matrix.
    """
    plot_matrix = matrix.copy()
    
    if np.allclose(np.triu(plot_matrix, k=1), 0):
        plot_matrix = plot_matrix + plot_matrix.T - np.diag(np.diag(plot_matrix))
    
    # Force the diagonal elements to be exactly 1.0 (self-correlation)
    np.fill_diagonal(plot_matrix, 1.0)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(plot_matrix, vmin=0, vmax=1, cmap="RdBu_r")
    
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

def compute_overall_coherence(epochs_path):
    print(f"Loading epochs from: {epochs_path.name}")
    epochs = mne.read_epochs(epochs_path, preload=True)
    
    # Pick only EEG channels
    epochs_eeg = epochs.copy().pick("eeg")
    ch_names = epochs_eeg.ch_names
    sfreq = epochs_eeg.info["sfreq"]
    
    print("  Computing OVERALL coherence (0.1 - 70.0 Hz)...")
    
    # Compute a single consolidated connectivity map
    con = spectral_connectivity_epochs(
        epochs_eeg,
        method="coh",        # Or 'imcoh' to reduce volume conduction
        mode="multitaper",
        sfreq=sfreq,
        fmin=2.5,            # Lower boundary of filtered data (change to 2.5 for wake and 0.5 for sleep)
        fmax=70.0,           # Upper boundary of your filtered data
        faverage=True,       # CRITICAL: Averages all frequencies into 1 single value per channel pair
        verbose=False
    )
    
    # Extract dense matrix and squeeze out the frequency dimension
    matrix_2d = np.squeeze(con.get_data(output="dense"))
    
    return matrix_2d, ch_names

def main():
    data_dir = Path("C:/Users/...") # Update to your path
    output_dir = data_dir / "connectivity_results"
    output_dir.mkdir(exist_ok=True)

    with open(data_dir / "subjects.txt", "r") as f:
        subjects = [line.strip() for line in f if line.strip()]

    for subject in subjects:
        epochs_path = data_dir / f"{subject}_preprocessed_epo.fif"
        
        if not epochs_path.exists():
            continue
            
        try:
            # 1. Compute the single overall matrix
            matrix, ch_names = compute_overall_coherence(epochs_path)
            
            # 2. Save numerical data
            np.save(output_dir / f"{subject}_overall_coherence.npy", matrix)
            
            # 3. Plot and save the single heatmap
            img_filename = output_dir / f"{subject}_overall_coherence.png"
            plot_connectivity_matrix(
                matrix, 
                ch_names, 
                title=f"{subject} - Overall Coherence (0.1-70 Hz)", 
                save_path=img_filename
            )
            print(f"Successfully saved overall connectivity for {subject}!")
            
        except Exception as e:
            print(f"Failed for {subject}: {e}")

if __name__ == "__main__":
    main()