import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def compute_grand_average(folder_path):
    """
    Loads all .npy files in a folder and computes their mathematical average.
    """
    folder = Path(folder_path)
    npy_files = list(folder.glob("*.npy"))
    
    if not npy_files:
        print(f"No .npy files found in {folder_path}")
        return None
        
    print(f"Averaging {len(npy_files)} matrices from {folder.name}...")
    
    matrices = []
    for file in npy_files:
        matrix = np.load(file)
        matrices.append(matrix)
        
    # Stack along a new axis and take the mean across subjects/runs
    grand_average = np.mean(np.array(matrices), axis=0)
    return grand_average

def plot_connectivity_matrix(matrix, ch_names, title, save_path=None):
    """
    Plots a complete, fully mirrored connectivity matrix.
    """
    plot_matrix = matrix.copy()
    
    # THE FIX: If the upper triangle is full of zeros, mirror the lower half onto it
    if np.allclose(np.triu(plot_matrix, k=1), 0):
        plot_matrix = plot_matrix + plot_matrix.T - np.diag(np.diag(plot_matrix))
    
    # Force the diagonal elements to be exactly 1.0 (self-correlation)
    np.fill_diagonal(plot_matrix, 1.0)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Now it will render the full square beautifully
    im = ax.imshow(plot_matrix, vmin=0, vmax=1, cmap="RdBu_r")
    
    # Keep the rest of your original styling code below...
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(np.arange(len(ch_names)))
    ax.set_yticks(np.arange(len(ch_names)))
    ax.set_xticklabels(ch_names, rotation=90, fontsize=8)
    ax.set_yticklabels(ch_names, fontsize=8)
    
    fig.colorbar(im, ax=ax, shrink=0.8, label="Coherence")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def main():
    # Base project directory
    base_dir = Path(r"C:/Users/...") # put your path here 
    
    sleep_folder = base_dir / "Sleep"
    wake_folder = base_dir / "wake"
    output_dir = base_dir / "grand_averages"
    output_dir.mkdir(exist_ok=True)
    
    # Try to load channel names to label the plot properly
    # Using your previously generated channel order file if it exists
    channel_file = base_dir / "channel_order.txt"
    if channel_file.exists():
        with open(channel_file, "r") as f:
            ch_names = [line.strip() for line in f if line.strip()]
    else:
        # Fallback: generating dummy labels if file isn't found
        print("channel_order.txt not found. Using generic indices for labels.")
        ch_names = [f"Ch{i}" for i in range(30)] # Assuming 30 channels based on your data output

    # 1. Compute and plot Sleep Grand Average
    sleep_avg = compute_grand_average(sleep_folder)
    if sleep_avg is not None:
        np.save(output_dir / "grand_avg_sleep_coherence.npy", sleep_avg)
        plot_connectivity_matrix(
            sleep_avg, 
            ch_names, 
            title="Grand Average Sleep Connectivity (Overall Coherence)", 
            save_path=output_dir / "grand_avg_sleep_coherence.png"
        )

    # 2. Compute and plot Wake Grand Average
    wake_avg = compute_grand_average(wake_folder)
    if wake_avg is not None:
        np.save(output_dir / "grand_avg_wake_coherence.npy", wake_avg)
        plot_connectivity_matrix(
            wake_avg, 
            ch_names, 
            title="Grand Average Wake Connectivity (Overall Coherence)", 
            save_path=output_dir / "grand_avg_wake_coherence.png"
        )

    print("\nGrand average processing complete!")

if __name__ == "__main__":
    main()