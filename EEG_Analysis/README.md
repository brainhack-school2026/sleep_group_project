The EEG Data was acquired simultaneously with fMRI, and therefore contains massive artifacts in the data. 

### **Processing** is divided into two parts:

A) **Removal of fMRI related artifacts** was done in MATLAB with the AMRI Toolbox which can be found here (https://www.amri.ninds.nih.gov/software.html). 
  - This toolbox was used as it was described in this paper by the authors of the dataset (https://www.sciencedirect.com/science/article/pii/S1053811911012067?via%3Dihub#s0010).
  - Our code is 'process_run_nomotion.m'.  
  - Step 1) is to remove the Gradient Artifact 
  - Step 2) is to remove the pulse artifact (ballistocardiogram (BCG) artifact)
  - The toolbox provided code to start with, we added extra code to process all participants at once. 
  - This code also uses the EEGLAB toolbox (https://sccn.ucsd.edu/eeglab/).
  - The gradient artifact removal function in AMRI 'amri_eeg_gac' automatically downsamples to 250 and lowpasses at 125 Hz after correction, but this can be set manually in the function call.

B) The rest of **standard preprocessing** steps are done in MNE Python. 
  - This script "EEG_post_MRI_processing_2.py" iterates through each subject and:
  	- trims the beginning and end 
  	- auto-rejects bad channels 
  	- bandpass filters the data 
  	- use ICA with EOG data to remove noise and eyeblinks 
  	- epochs the data either into 30s intervals for sleep data, or 2s for wake (must be specified). 


### **EEG Analysis** is was done by: 

C) Calculating correlation matrices with the coherence value for each participant run. 
  - 'plot_connectivityMatrices_perFrequency.py' calculates the coherence matrix for each run PER frequency band (alpha, beta, delta, theta)
  - 'plot_connectivityMatrices_FREQUENCIES.py' takes the above matrices per state and frequency and calculates an overall mean matrix for all participants. (Sleep frequencies and wake frequencies)  
  - 'plot_connectivityMatrices_OVERALL.py' calculates the coherence matrix for each run with no specified frequency (overall coherence) 
  - 'Average_Connectivity_Matrices.py' takes the above matrices and calculates a grand average matrix with all participants in sleep and in wake state. 

D) Linear Mixed Effects Model for analysis 
  - 'LinearMixedModel_EEG.ipynb' is a notebook that does the following: 
  - A linear mixed effects model is run on the mean coherence values for each run per state and frequency to determine if there are any statistically significant differences between the wake vs sleep states and between the frequency bands of each. 
