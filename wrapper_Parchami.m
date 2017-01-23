function wrapper_Parchami(inDir,outDir,M)

% Inputs:
%   inDir: Input Directory
%   outDir: Output Directory
%   M : number of past frames using for smoothing scheme(equation 9 of[1])

% Outputs:
%   y: output audio after MVDR beamforming. The noise covariance matrix
%   estimation is based on [1]

% [1]Parchami, M., Zhu, W. P., & Champagne, B. (2015, May).
% A new algorithm for noise PSD matrix estimation in multi-microphone
% speech enhancement based on recursive smoothing.

mDir = strcat('M_',int2str(M))
inpath_split = strsplit(inDir,'/')
mkdir(outDir,char(strcat('Parchami/',mDir,'/', inpath_split(end),'/wavout')))
mkdir(outDir,char(strcat('Parchami/',mDir,'/', inpath_split(end),'/data')))
% take component of input path,for example if Input is /Users/chime3/tr05_caf_simu'
% then inpath_split are 'Users' 'chime3' 'tr05_caf_simu'
wav_outDir = fullfile(outDir,char(strcat('Parchami/',mDir,'/', inpath_split(end),'/wavout')));
data_outDir = fullfile(outDir,char(strcat('Parchami/',mDir,'/', inpath_split(end),'/data')));
% inFilesOrPattern = '.*\.CH1\.wav$';
% search file in input directory, inFiles is a matrix contains file names such as bus.wav
inFilesOrPattern = '.*\.CH1\.wav';
inFiles = findFiles(inDir, inFilesOrPattern); % inFiles look like is m x 1 cell, example 1 cell: F02_01BO030M_CAF.CH0.wav
for h = 1:length(inFiles);
    inFile = fullfile(inDir, inFiles{h});
    inFileNoCh = strrep(inFiles{h},'.CH1','');
    outWavFile = fullfile(wav_outDir, inFileNoCh);
    [inD inF inE] = fileparts(inFile);
    info = audioinfo(inFile);
    fs = info.SampleRate;
    x = zeros(info.TotalSamples,1,6);
    for i = 1 : 6
        chanFile = fullfile(inD, [strrep(inF, '.CH1', sprintf('.CH%d', i)) inE]);
        noisy_sampling = audioread(chanFile);
        x(:,:,i) = noisy_sampling;
    end
    outMaskFile = fullfile(data_outDir, strrep(inFileNoCh, '.wav', '.mat'));
    nsample = size(noisy_sampling,1);
    x = permute(x,[3 1 2]);
    X = stft_multi(x,1024);
    
    Y = mvdrParchami(X,M);
    
    y = istft_multi(Y,nsample);
    audiowrite(outWavFile,y',fs);
end