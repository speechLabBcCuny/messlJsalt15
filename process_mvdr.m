% Use this script to create masks for the CHiMe3 data. For each 7 channel
% utterance, it will create a clean audio wav file and compute the masks
% between the clean audio and each of 7 channel utterance.

% add the library
addpath('../mimlib');
addpath('../utils');

% working direction
% contain wav files for each channel for one 
workDir = '/Users/Near/Desktop/MESSL/mvdr_test/dev/';
% Calculate delay between CH0 and the others
corrDir = strcat(workDir,'/corr/'); %temporary
corrDataDir = strcat(corrDir,'/data/');
% directory to save cleaned (MVDR) wav
outDir = strcat(workDir,'/out/'); %temporary
% directory to save the masks
outMaskDir = strcat(workDir,'/mask/');

if exist(corrDir,'dir')==7 || exist(outDir,'dir')==7
    error('Temporary directories should not already exist');
end

try
% Find the correlations of CH0 with CH* files
enhance_wrapper(@stubI_justXcorr, workDir, corrDir, [1 1], 0, 0, 2, '.CH1');

% Fix the wav files with delay
correctCh0Delay(workDir, corrDataDir);

% Create cleaned audio wav
enhance_wrapper(@(X, fail, fs, file) stubI_supervisedMvdr(X, fail, fs, file, 0.75, 0.85, 15), ... 
    workDir, outDir, [1 1], 1, 0, 2, '.CH1'); 
end

