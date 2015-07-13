function enhance_wrapper(stubFn, inDir, outDir, part, overwrite, ignoreErrors, filePerChan, inFiles)

% Wrapper like for CHiME3, but working with arbitrary multichannel wavsxs
%
% For compatibility with CHiME3 wrapper, takes as argument an enhancement
% stub function with the following interface:
%
% [Y data] = stubFn(X, fail, sr, fileName);
%
% Y is the estimated single channel spectrogram of the speech, X is the
% multi-channel spectrogram of the noisy speech, N is the multi-channel
% spectrogram of up to 5 seconds of noise preceeding the speech, Ncov is
% the computed frequency-dependent noise covariance matrix, fail is a
% binary vector indicating whether each mic has failed, TDOA is a matrix of
% TDOA estimates for each channel for each frame measured in seconds, and
% sr is the sampling rate.
%
% Data will be written in a standard directory structure rooted at outDir.
% Output wav files in the wav/ subdirectory and data as mat files in the
% data/ subdirectory.
% 
% Part is a tuple [n N] meaning process the nth of N sets of utterances to
% allow for easy parallelization across matlab sessions.

if ~exist('overwrite', 'var') || isempty(overwrite), overwrite = false; end
if ~exist('part', 'var') || isempty(part), part = [1 1]; end
if ~exist('ignoreErrors', 'var') || isempty(ignoreErrors), ignoreErrors = false; end
if ~exist('filePerChan', 'var') || isempty(filePerChan), filePerChan = false; end
if ~exist('inFiles', 'var'), inFiles = []; end

% Define hyper-parameters
pow_thresh=-20; % threshold in dB below which a microphone is considered to fail
wlen = 1024; % STFT window length

if strcmp(inDir, outDir)
    error('Not overwriting input: %s == %s', inDir, outDir);
end

if isempty(inFiles)
    if filePerChan
        inFiles = findFiles(inDir, '(real|simu).*\.CH1\.wav');
    else
        inFiles = findFiles(inDir, '.*.wav');
    end
    
    % Shuffle file list reproducibly
    inFiles = inFiles(runWithRandomSeed(22, @randperm, length(inFiles)));
end

if filePerChan
    lastNChan = [];
end

for f = part(1):part(2):length(inFiles)
    inFile = fullfile(inDir, inFiles{f});
    inFileNoCh = strrep(inFiles{f}, '.CH1', '');
    outWavFile = fullfile(outDir, 'wav', inFileNoCh);
    outMaskFile = fullfile(outDir, 'data', strrep(inFileNoCh, '.wav', '.mat'));
    
    if exist(outWavFile, 'file') && ~overwrite
        fprintf('%d: Skipping %s\n', f, outWavFile);
        continue
    else
        fprintf('%d: %s\n', f, outWavFile);
    end
    
    % Read file
    if filePerChan
        [inD inF] = fileparts(inFile);
        assert(reMatch(inF, '\.CH1'));
        glob = strrep(inF, '.CH1', '.CH[1-9]');
        [chanFilesShort,chanFiles] = findFiles(inD, glob);

        if ~isempty(lastNChan) && (length(chanFiles) ~= lastNChan)
            warning('Messl:Chime3:NChanChanged', 'Number of channels changed from %d to %d: %s', ...
                lastNChan, length(chanFiles), join(chanFilesShort , ', '));
        end
        lastNChan = length(chanFiles);
        
        [sz fs] = wavread(chanFiles{1}, 'size');
        x = zeros(sz(1), length(chanFiles));
        for i = 1:length(chanFiles)
            [x(:,i) fsi] = wavread(chanFiles{i});
            assert(fsi == fs);
        end
    else
        [x fs] = wavread(inFile);
    end
    nsampl = size(x,1);

    % Determine if any mics have failed
    xpow=sum(x.^2,1);
    xpow=10*log10(xpow/max(xpow));
    fail=(xpow<=pow_thresh);

    % STFT
    X = stft_multi(x.',wlen);
    [nbin,nfram,~] = size(X);

    %%% Call the stub
    try
        [Y data] = stubFn(X, fail, fs, inFiles{f});
    catch ex
        if ignoreErrors
            disp(getReport(ex))
            Y = X;
            data = [];
        else
            rethrow(ex)
        end
    end
    data = structCast(data, @isnumeric, @single);
    
    % Inverse STFT and write WAV file with one source per channel
    y = istft_multi(Y, nsampl).';
    y = y * 0.999/max(abs(y(:)));
    ensureDirExists(outWavFile);
    wavwrite(y, fs, outWavFile);
    ensureDirExists(outMaskFile);
    save(outMaskFile, 'data', 'fs', 'nbin', 'nfram', 'nsampl', 'fail');
end
