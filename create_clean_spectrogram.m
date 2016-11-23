% Use this script to create masks for the CHiMe3 data. For each 7 channel
% utterance, it will create a clean audio wav file and compute the masks
% between the clean audio and each of 7 channel utterance.

% add the library
addpath('../mimlib');
addpath('../utils');

% working direction
% contain wav files for each channel for one 
workDir = '/home/data/CHiME3/data/audio/16kHz/isolated/';
% directory to save cleaned wav
wavDir = '/home/data/CHiME3/data/audio/16kHz/local/messl-mvdr-output';

outDir = '/home/data/CHiME3/data/audio/16kHz/local/clean-spect/'; %temporary
try
% compute and save masks based on cleaned audio
enhance_wrapper(@(X,fail,fs,file) stubI_Clean(file, wavDir), ...
    workDir, outDir, [1 1], 1, 0, 2, '.*05.*real.*\.CH1\.wav'); 
catch
    %remove temporary folders and files
    error('Error during mask creation');
end  