function [Y data mask Xp] = stubI_LSTMMessl3(X, fail, fs, inFile, I, refMic, d, useHardMask, beamformer, varargin, lstm_dir)

% Multichannel MESSL mask with simple beamforming initialized from cross
% correlations between mics.

if ~exist('I', 'var') || isempty(I), I = 1; end
if ~exist('refMic', 'var') || isempty(refMic), refMic = 0; end
if ~exist('d', 'var') || isempty(d), d = 0.35; end
if ~exist('useHardMask', 'var') || isempty(useHardMask), useHardMask = true; end
if ~exist('beamformer', 'var') || isempty(beamformer), beamformer = 'bestMic'; end

% Check that mrfHardCompatExp is not zero
ind = find(strcmp(varargin, 'mrfHardCompatExp'));
if useHardMask && (isempty(ind) || (varargin{ind+1} == 0))
    error('Must set "mrfHardCompatExp" to nonzero value with useHardMask')
end

%inFile is the filename for the auido file, so make use of it to find LSTM mask file.
lstm_file = fullfile(lstm_dir,inFile)
% six masks are already combined in one single mask in python
%load the lstm mask
LSTM_Mask = load(lstm_file);


maxSup_db = -40;

maxSup = 10^(maxSup_db/20);
tau = tauGrid(d, fs, 31);
fprintf('Max ITD: %g samples\n', tau(end));

maskInit = LSTM_Mask;

% See if reference mic has failed. refMic = 0 means all pairs.
if (refMic > 0) && fail(refMic)
    refMic = find(~fail, 1, 'first');
    if isempty(refMic)
        error('All potential reference mics have failed')
    end
end

% MESSL for mask
messlOpts = [{'GarbageSrc', 1, 'fixIPriors', 1, 'maskInit', maskInit, 'refMic', refMic, 'maskHold', Inf} varargin];
[p_lr_iwt params hardMasks] = messlMultichannel(X(2:end-1,:,~fail), tau, I, messlOpts{:});

if useHardMask
    mask = squeeze(hardMasks(1,:,:,:));
else
    mask = prob2mask(squeeze(p_lr_iwt(1,:,:,:)));
end

z = zeros([1 size(X,2) size(mask,3)]);
mask = cat(1, z, mask, z);
mask = maxSup + (1 - 2*maxSup) * mask;

switch beamformer
    case 'bestMic'
        Xp = pickChanWithBestSnr(X, mask, fail);
    case 'mvdr'
        [Xp mvdrMask mask] = maskDrivenMvdrMulti(X, mask, fail, params.perMicTdoa);
        data.mvdrMask = single(mvdrMask);
    case 'souden'
        [Xp mvdrMask mask] = mvdrSoudenMulti(X, mask, fail);
        data.mvdrMask = single(mvdrMask);
    otherwise
        error('Unknown beamformer: %s', beamformer)
end

data.mask = single(mask);
data.params = params;

% Output spectrogram(s)
Y = Xp .* mask;