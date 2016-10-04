% add the library
addpath('../mimlib');
addpath('../utils');
% working direction
% contain wav files for each channel for one 
workDir = '/Users/Near/Desktop/MESSL/mvdr_test/dev2/';
% Calculate delay between CH0 and the others
corrDir = strcat(workDir,'/corr/');
corrDataDir = strcat(corrDir,'/data/');
% directory to save cleaned (MVDR) wav
outDir = strcat(workDir,'/out/');
% directory to save the masks
outMaskDir = strcat(workDir,'/mask/');

% Find the correlations of CH0 with CH* files
enhance_wrapper(@stubI_justXcorr, workDir, corrDir, [1 1], 0, 0, 2, '.CH1');

% Fix the wav files with delay
correctCh0Delay(workDir, corrDataDir);

% Create cleaned audio wav
enhance_wrapper(@(X, fail, fs, file) stubI_supervisedMvdr(X, fail, fs, file, 0.75, 0.85, 15), ... 
    workDir, outDir, [1 1], 1, 0, 2, '.CH1'); 

% compute and save masks based on cleaned audio
enhance_wrapper(@(X,fail,fs,file) stubI_Masks(X, fail, fs, file, outDir, 'ideal_amplitude', 0.75, 0.85, 15), ...
    workDir, outMaskDir, [1 1], 1, 0, 2, '.CH1'); 

rmdir(corrDir);
rmdir(outDir);