import mne
import os
import numpy as np
import mne
from pathlib import Path
from mne.preprocessing import ICA
from pyprep import NoisyChannels

def auto_reject_channels(raw): 
    raw_copy = raw.copy().pick("eeg")
    
    try: 
        raw_copy.set_montage("standard_1020")
    except ValueError as e: 
        print(f"Montage Warning: Some channels didn't match standard 10/20. Trying with match_case=False ... Error: {e}")
        raw_copy.set_montage("standard_1020", match_case=False, on_missing="warn")

    # find noisy channels 
    nd = NoisyChannels(raw_copy, random_state=97)
    nd.find_all_bads() 

    # Extract the list of bad channel names 
    bad_channels = nd.get_bads() 
    print(f"PyPrep identified bad channels: {bad_channels}")

    return bad_channels

def run_ica_pipeline(raw): 
    # 1. Explicitly set the montage on the main raw object first
    # This gives MNE the 3D coordinates it needs to interpolate bad channels
    try:
        raw.set_montage("standard_1020", match_case=False, on_missing="warn")
    except Exception as e:
        print(f"Main Montage Warning: Could not assign standard 1020 layout: {e}")

    # 2. Backup copy
    raw_original = raw.copy()

    # 3. Mark and interpolate bad channels on the working dataset
    bad_channels = auto_reject_channels(raw)
    raw.info['bads'] = bad_channels
    
    # Interpolate bads using spherical splines 
    print(f"Interpolating bad channels: {bad_channels}...")
    raw.interpolate_bads(reset_bads=True, on_bad_position='ignore')

    # 4. Bandpass data (0.1-70 Hz)  
    print("Preparing data with a 0.1 - 70 Hz band-pass filter for sleep...")
    raw_clean = raw.copy() 
    raw_clean.filter(l_freq=0.1, h_freq=70.0, picks="eeg")
    raw_clean.notch_filter(freqs=[60], picks="eeg")

    # 5. Fit ICA  
    print("Fitting ICA...")
    ica = ICA(
        n_components=0.99, 
        method='fastica', 
        random_state=97,
        max_iter="auto"
    )
    ica.fit(raw_clean, picks="eeg")

    # 6. Detect eye blinks using ICA copy 
    eog_indices = []
    for ch in ["Fp1", "Fp2"]:
        if ch in raw_clean.ch_names:
            inds, scores = ica.find_bads_eog(raw_clean, ch_name=ch)
            eog_indices.extend(inds)
    
    eog_indices = list(set(eog_indices))
    ica.exclude = eog_indices
    print("ICA components selected for removal:", eog_indices)

    # 7. Apply the ICA weights 
    print("Applying ICA weights to the data...")
    ica.apply(raw_clean)

    # 8. Apply average reference 
    raw_clean.set_eeg_reference(ref_channels="average", projection=False)

    print("Processing complete.")

    # Return the clean data and the true unmodified original
    return raw_clean, raw_original

def epoching(raw_clean, duration=30): # 30 second for sleep runs, 2 second for wake runs
    print(f"\nEpoching data into {duration}-second windows...")
    
    # define thresholds: 200-300 microvolts (to capture k-complexes in N2 sleep stages)
    reject_criteria = dict(eeg=300e-6)

    overlap_duration = 20

    # create fixed length periods
    epochs = mne.make_fixed_length_epochs(
        raw_clean, 
        duration=duration,
        overlap=overlap_duration,
        preload=True,
        reject_by_annotation=True
    )

    # drop epochs that exceed peak-to-peak amlitude threholds
    print("Automatically dropping high-amplitude epochs...")
    epochs.drop_bad(reject=reject_criteria)
    
    # Print out summary statistics
    print(f"Total epochs generated: {len(epochs) + len(epochs.drop_log) if epochs.drop_log else len(epochs)}")
    print(f"Epochs retained after cleaning: {len(epochs)}")
    
    return epochs
    

def compare_traces(epochs_clean, raw_original):
    # compare before and after of post MRI and post artifact filtered traces

    start_sec = 100       # change this if this region is flat
    duration_sec = 40

    print("BEFORE preprocessing:")
    raw_original.plot(
        start=start_sec,
        duration=duration_sec,
        n_channels=30,
        scalings=dict(eeg=100e-6),
        block=True
    )

    print("AFTER preprocessing:")
    epochs_clean.plot(n_channels=30, scalings=dict(eeg=100e-6), block=True)


def compare_psd(raw_clean, raw_original):
    # compare power spectrum before and after preprocessing
    import matplotlib.pyplot as plt
    print("BEFORE preprocessing PSD:")
    fig_before = raw_original.compute_psd(
        fmax=125,
        picks="eeg"
    ).plot(
        average=True,
        amplitude=False,
        show=False
    )

    plt.show()

    print("AFTER preprocessing PSD:")
    fig_after = raw_clean.compute_psd(
        fmax=125,
        picks="eeg"
    ).plot(
        average=True,
        amplitude=False,
        show=False
    )

    plt.show()

    print("PSD check complete.")


def main(): 
    subjects_file = Path("subjects.txt")

    if not subjects_file.exists(): 
        print(f"Error: The file {subjects_file} was not found in current directory.")
        return 
    
    with open(subjects_file, "r") as f: 
        # .strip() removes hidden newlines (\n) 
        subjects = [line.strip() for line in f if line.strip()]
        
    for subject in subjects: 
        # Path to your BrainVision .vhdr files
        vhdr_path = Path(f"/{subject}.vhdr")
        
        if not vhdr_path.exists(): 
            print(f"\n WARNING: File not found for subject '{subject}'. Skipping...")
            continue 
        try:    
            print(f"\nProcessing subject: {subject}")

            # Load the file
            raw = mne.io.read_raw_brainvision(vhdr_path, preload=True)

            # --- Trim 26 seconds from the start and end ---
            start_trim = 26.0 
            end_trim = raw.times[-1] - 10.0 

            print(f"Original duration: {raw.times[-1]:.2f} seconds")
            raw.crop(tmin=start_trim, tmax=end_trim)
            print(f"New cropped duration: {raw.times[-1]:.2f} seconds")

            # Print basic information
            print(f"\nSampling frequency: {raw.info['sfreq']} Hz.")
            print(f"\nNumber of channels: {raw.info['nchan']}")
            print(f"\nRecording duration in minutes: {raw.times[-1] / 60}")

            # preprocess with ICA 
            raw_clean, raw_original = run_ica_pipeline(raw)

            # convert continuous clean data to 30s or 2s epochs 
            epochs_clean = epoching(raw_clean, duration=30) # 30 for sleep, 2 for wake 
        
            # save the preprocessed file as a new file
            output_name = f"{subject}_preprocessed.fif"
            raw_clean.save(output_name, overwrite=True)
            print(f"Saved preprocessed file as {output_name}.")

            # save the epoched data 
            epoch_output_name = f"{subject}_preprocessed_epo.fif"
            epochs_clean.save(epoch_output_name, overwrite=True)
            print(f"Saved preprocessed epochs as {epoch_output_name}.")

            # Visualization
            # compare_traces(epochs_clean, raw_original)
            # compare_psd(raw_clean, raw_original)

            print(f"Finished processing subject: {subject}\n" + "-"*40)

        except Exception as e: 
            print(f"Error processing subject {subject}: {e}")
            continue 

    print("\nAll available subjects processed successfully!")

if __name__ == "__main__": 
    main()