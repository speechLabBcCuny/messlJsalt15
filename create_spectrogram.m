% Use this script to create spectrogram for the CHiMe3 data. 

% add the library
addpath('../mimlib');
addpath('../utils');

% working direction
% contain wav files for each channel for one 
workDir = '/home/data/CHiME3/data/audio/16kHz/isolated/';
% directory to save the spectrogram
outInputDir = strcat(workDir,'/spectrogram/');
try
% compute and save spectrogram for CH1 to CH6
enhance_wrapper(@(X,fail,fs,file) stubI_Spectrogram(X), ...
    workDir, outInputDir, [1 1], 1, 0, 2, '[de]t05.*real.*\.CH1\.wav$');
end
