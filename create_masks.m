% Use this script to create masks for the CHiMe3 data. For each 7 channel
% utterance, it will create a clean audio wav file and compute the masks
% between the clean audio and each of 7 channel utterance.

% add the library
addpath('../mimlib');
addpath('../utils');

% working direction
% contain wav files for each channel for one 
workDir = '/Users/Near/Desktop/MESSL/mvdr_test/dev2/';
% Calculate delay between CH0 and the others
corrDir = strcat(workDir,'/corr/'); %temporary
corrDataDir = strcat(corrDir,'/data/');
% directory to save cleaned (MVDR) wav
outDir = strcat(workDir,'/out/'); %temporary
% directory to save the masks
outMaskDir = strcat(workDir,'/mask/');

mode = 'ideal_complex';
try
% compute and save masks based on cleaned audio
enhance_wrapper(@(X,fail,fs,file) stubI_Masks(X, file, outDir, mode), ...
    workDir, strcat(outMaskDir,'/',mode,'/'), [1 1], 1, 0, 2, '.CH1'); 
catch
    %remove temporary folders and files
    error('Error during mask creation');
end  

%remove temporary folders and files
%rmdir(corrDir,'s');
%rmdir(outDir,'s');