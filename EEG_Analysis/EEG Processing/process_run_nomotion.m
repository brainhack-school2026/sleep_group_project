function process_run_nomotion(raw_vhdr_path, output_prefix)
    try
        fprintf('--- Starting MATLAB Pipeline for: %s ---\n', raw_vhdr_path);
        
        %% 1. Environment Setup & Data Loading
        % Dynamically locate EEGLAB on the cluster path
        eeglab_root = fileparts(which('eeglab'));
        if isempty(eeglab_root)
            error('EEGLAB is not found on the MATLAB path. Please check your startup.m or cluster paths.');
        end
        addpath(genpath(eeglab_root));

        % Load raw dataset
        [input_dir, file_name, ~] = fileparts(raw_vhdr_path);
        eeg = pop_loadbv(input_dir, [file_name, '.vhdr']);

        %% 2. Remove Gradient Artifacts, Low-Pass Filter, and Downsample
        fprintf('--- Pipeline: Removing GA, Low-passing at 125Hz, and Downsampling to 250Hz... ---\n');
        eeg_gac = amri_eeg_gac(eeg, 'lowpass.cutoff', 125, 'downsample', 250);

        %% 3. Prepare for Pulse Artifact Removal
        ecg_gac = pop_select(eeg_gac, 'channel', {'ECG'});
        eeg_gac_r = amri_eeg_rpeak(eeg_gac, ecg_gac);
        eeg_for_ica = pop_select(eeg_gac_r, 'nochannel', {'ECG', 'EOG'});

        %% 4. Remove Pulse Artifacts via ICA (At 250 Hz)
        eeg_clean = amri_eeg_cbc(eeg_for_ica, ecg_gac);

        %% 5. Save BrainVision Triplet (.vhdr, .vmrk, .dat)
        fprintf('--- Pipeline: Exporting clean files to %s... ---\n', output_prefix);
        [out_dir, ~, ~] = fileparts(output_prefix);
        if ~exist(out_dir, 'dir'), mkdir(out_dir); end
        
        pop_writebva(eeg_clean, output_prefix, 'DataOrientation', 'MULTIPLEXED');
        fprintf('--- Job successfully finished processing! ---\n');
        
    catch ME
        fprintf('ERROR EXECUTION FAILED: %s\n', ME.message);
        exit(1); % Force failure exit status so cluster logs register the crash
    end
    exit(0);
end