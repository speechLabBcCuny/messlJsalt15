function enhance_wrapper(stubFn, inDir, outDir, part, overwrite, ignoreErrors)

% Wrapper like for CHiME3, but working with arbitrary multichannel wavsxs
%
% For compatibility with CHiME3 wrapper, takes as argument an enhancement
% stub function with the following interface:
%
% Y = stubFn(X, N, Ncov, fail, TDOA_s, sr);
%
% BUT note that N, Ncov, and TDOA_s will all be empty. Y is the estimated
% single channel spectrogram of the speech, X is the multi-channel
% spectrogram of the noisy speech, N is the multi-channel spectrogram of up
% to 5 seconds of noise preceeding the speech, Ncov is the computed
% frequency-dependent noise covariance matrix, fail is a binary vector
% indicating whether each mic has failed, TDOA is a matrix of TDOA
% estimates for each channel for each frame measured in seconds, and sr is
% the sampling rate.
%
% Data will be written in a standard directory structure rooted at outDir.
% 
% Part is a tuple [n N] meaning process the nth of N sets of utterances to
% allow for easy parallelization across matlab sessions.

if ~exist('overwrite', 'var') || isempty(overwrite), overwrite = false; end
if ~exist('part', 'var') || isempty(part), part = [1 1]; end
if ~exist('ignoreErrors', 'var') || isempty(ignoreErrors), ignoreErrors = false; end

% Define hyper-parameters
pow_thresh=-20; % threshold in dB below which a microphone is considered to fail
wlen = 1024; % STFT window length

if strcmp(inDir, outDir)
    error('Not overwriting input: %s == %s', inDir, outDir);
end

inFiles = findFiles(inDir, '.*.wav');

for f = part(1):part(2):length(inFiles)
    inFile = fullfile(inDir, inFiles{f});
    outFile = fullfile(outDir, inFiles{f});
    
    if exist(outFile, 'file') && ~overwrite
        fprintf('%d: Skipping %s\n', f, outFile);
        continue
    else
        fprintf('%d: %s\n', f, outFile);
    end
    
    % Read file
    [x fs] = wavread(inFile);
    nsampl = size(x,1);

    % Determine if any mics have failed
    xpow=sum(x.^2,1);
    xpow=10*log10(xpow/max(xpow));
    fail=(xpow<=pow_thresh);

    % STFT
    X = stft_multi(x.',wlen);
    [nbin,nfram,~] = size(X);

    % Setup dummy inputs
    N = [];
    Ncov = [];
    TDOA = [];
    
    %%% Call the stub
    try
        Y = stubFn(X, N, Ncov, fail, TDOA, fs);
    catch ex
        if ignoreErrors
            disp(getReport(ex))
            Y = ones(nbin, nfram);
        else
            rethrow(ex)
        end
    end
    
    % Inverse STFT and write WAV file
    y = istft_multi(Y,nsampl).';
    y = y * 0.999/max(abs(y));
    ensureDirExists(outFile);
    wavwrite(y, fs, outFile);
end
