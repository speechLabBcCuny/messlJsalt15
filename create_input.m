% Use this script to create spectrogram for the CHiMe3 data. 

% add the library
addpath('../mimlib');
addpath('../utils');

% working direction
% contain wav files for each channel for one 
workDir = '/Users/Near/Desktop/MESSL/mvdr_test/dev2/';
% directory to save the masks
outInputDir = strcat(workDir,'/input/');
try
% compute and save masks based on cleaned audio
enhance_wrapper(@(X,fail,fs,file) createInput(X), ...
    workDir, outInputDir, [1 1], 1, 0, 2, '.CH1'); 
catch
    error('Error during mask creation');
end  