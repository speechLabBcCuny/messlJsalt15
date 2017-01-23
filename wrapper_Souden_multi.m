function wrapper_Souden_multi(inDir,outDir)
% Inputs:
%   inDir: Input Directory
%   outDir: Output Directory
%   M : number of past frames using for smoothing scheme(equation 9 of[1])

% Outputs:
%   y: output audio after MVDR beamforming. The noise covariance matrix
%   estimation is based on [1]

% [1]Souden, M., Chen, J., Benesty, J., & Affes, S. (2011).
%  An integrated solution for online multichannel noise tracking and reduction.
% IEEE Transactions on Audio, Speech, and Language Processing, 19(7), 2159-2169.


inpath_split = strsplit(inDir,'/');

mkdir(outDir,char(strcat('Souden/', inpath_split(end),'/wavout')));
mkdir(outDir,char(strcat('Souden/',inpath_split(end),'/data')));
wav_outDir = fullfile(outDir,char(strcat('Souden/', inpath_split(end),'/wavout')));
data_outDir = fullfile(outDir,char(strcat('Souden/',inpath_split(end),'/data')));

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
    
    for i = 1 : 6;
        chanFile = fullfile(inD, [strrep(inF, '.CH1', sprintf('.CH%d', i)) inE]);
        noisy_sampling = audioread(chanFile);
        x(:,:,i) = noisy_sampling;
    end
    
    outMaskFile = fullfile(data_outDir, strrep(inFileNoCh, '.wav', '.mat'));
    nsample = size(noisy_sampling,1);
    x = permute(x,[3 1 2]); % from nsample x 1 x C to C * nsample
    
    X = stft_multi(x,1024);
    %save(outMaskFile,'X');
    
    Y = mvdrSouden_multi(X);
    
    y = istft_multi(Y,nsample);
    audiowrite(outWavFile,y',fs);
end