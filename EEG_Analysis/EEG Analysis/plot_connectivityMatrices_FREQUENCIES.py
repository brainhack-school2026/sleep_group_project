import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

def extract_band_name(filename):
    """
    Extracts the band name (e.g., 'Delta', 'Theta') from a filename like:
    'sub-01_task-sleep_run-1_eeg_Delta_coherence.npy'
    """
    # Look for common band names in the filename
    for band in ["Delta", "Theta", "Alpha", "Beta", "Sigma", "Gamma"]:
        if band.lower() in filename.lower():
            return band
    return "Unknown"

def compute_band_averages(folder_path):
    """
    Groups .npy files by frequency band and computes the grand average for each.
    """
    folder = Path(folder_path)
    npy_files = list(folder.glob("*.npy"))
    
    if not npy_files:
        print(f" No .npy files found in {folder_path}")
        return {}
        
    # Group file paths by band
    band_groups = {}
    for file in npy_files:
        band = extract_band_name(file.name)
        if band == "Unknown":
            continue
        if band not in band_groups:
            band_groups[band] = []
        band_groups[band].append(file)
        
    # Compute averages
    band_averages = {}
    for band_name, files in band_groups.items():
        print(f"  Averaging {len(files)} matrices for the {band_name} band...")
        matrices = [np.load(f) for f in files]
        band_averages[band_name] = np.mean(np.array(matrices), axis=0)
        
    return band_averages

def plot_perfect_symmetric_matrix(matrix, ch_names, title, save_path=None):
    """
    Plots a complete matrix, explicitly 
    mirroring the lower triangle, and forces the diagonal to 1.0.
    """
    plot_matrix = matrix.copy()
    
    if np.allclose(np.triu(plot_matrix, k=1), 0):
        plot_matrix = plot_matrix + plot_matrix.T - np.diag(np.diag(plot_matrix))
        
    # Force the diagonal elements to be exactly 1.0 (self-correlation)
    np.fill_diagonal(plot_matrix, 1.0)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(plot_matrix, vmin=0, vmax=1, cmap="RdBu_r")
    
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(np.arange(len(ch_names)))
    ax.set_yticks(np.arange(len(ch_names)))
    ax.set_xticklabels(ch_names, rotation=90, fontsize=8)
    ax.set_yticklabels(ch_names, fontsize=8)
    
    # White grid lines
    ax.set_xticks(np.arange(-.5, len(ch_names), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(ch_names), 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', bottom=False, left=False)
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Coherence", fontsize=11, weight="bold")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def main():
    # Setup directories
    base_dir = Path(r"C:/Users/...") # Put your path here
    
    wake_freq_folder = base_dir / "wake_frequencies"
    sleep_freq_folder = base_dir / "sleep_frequencies"
    
    output_dir = base_dir / "frequency_grand_averages"
    output_dir.mkdir(exist_ok=True)
    
    # Load channel labels
    channel_file = base_dir / "channel_order.txt"
    if channel_file.exists():
        with open(channel_file, "r") as f:
            ch_names = [line.strip() for line in f if line.strip()]
    else:
        print("channel_order.txt not found. Using generic indices.")
        ch_names = [f"Ch{i}" for i in range(30)]

    # 1. Process Wake Frequencies
    print("Processing WAKE frequency bands...")
    wake_band_avgs = compute_band_averages(wake_freq_folder)
    for band_name, avg_matrix in wake_band_avgs.items():
        # Save array
        np.save(output_dir / f"grand_avg_wake_{band_name}.npy", avg_matrix)
        # Plot full symmetric view
        plot_perfect_symmetric_matrix(
            avg_matrix, 
            ch_names, 
            title=f"Grand Average Wake Connectivity - {band_name} Band", 
            save_path=output_dir / f"grand_avg_wake_{band_name}.png"
        )

    print("\n" + "="*40 + "\n")

    # 2. Process Sleep Frequencies
    print("Processing SLEEP frequency bands...")
    sleep_band_avgs = compute_band_averages(sleep_freq_folder)
    for band_name, avg_matrix in sleep_band_avgs.items():
        # Save array
        np.save(output_dir / f"grand_avg_sleep_{band_name}.npy", avg_matrix)
        # Plot full symmetric view
        plot_perfect_symmetric_matrix(
            avg_matrix, 
            ch_names, 
            title=f"Grand Average Sleep Connectivity - {band_name} Band", 
            save_path=output_dir / f"grand_avg_sleep_{band_name}.png"
        )

    print("\nAll frequency band grand averages computed and fully mirrored successfully!")

if __name__ == "__main__":
    main()