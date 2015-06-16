function [Y mask Xp] = stubI_messlMc(X, fail, fs, I, allPairs, d, useHardMask, varargin)

% Multichannel MESSL mask with simple beamforming initialized from cross
% correlations between mics.

if ~exist('I', 'var') || isempty(I), I = 1; end
if ~exist('allPairs', 'var') || isempty(allPairs), allPairs = true; end
if ~exist('d', 'var') || isempty(d), d = 0.35; end
if ~exist('useHardMask', 'var') || isempty(useHardMask), useHardMask = true; end

% Check that mrfHardCompatExp is not zero
ind = find(strcmp(varargin, 'mrfHardCompatExp'));
if useHardMask && (isempty(ind) || (varargin{ind+1} == 0))
    error('Must set "mrfHardCompatExp" to nonzero value with useHardMask')
end

maxSup_db = -40;

maxSup = 10^(maxSup_db/20);
tau = tauGrid(d, fs, 31);
fprintf('Max ITD: %g samples\n', tau(end));

maskInit = [];

% Figure out which mic to use for reference
if allPairs
    refMic = 0;
else
    if ~fail(2)
        refMic = 2;
    else
        refMic = find(~fail, 1, 'first');
    end
    if isempty(refMic)
        error('All potential reference mics have failed')
    end
end

% MESSL for mask
messlOpts = [{'GarbageSrc', 1, 'fixIPriors', 1, 'maskInit', maskInit, 'refMic', refMic} varargin];
[p_lr_iwt params hardMasks] = messlMultichannel(X(2:end-1,:,~fail), tau, I, messlOpts{:});

if useHardMask
    mask = squeeze(hardMasks(1,:,:,:));
else
    mask = prob2mask(squeeze(p_lr_iwt(1,:,:,:)));
end

z = zeros([1 size(X,2) size(mask,3)]);
mask = cat(1, z, mask, z);
mask = maxSup + (1 - maxSup) * mask;

% Figure out what to apply the mask to
% Stupidest way: pick the channel with best estimated SNR
P = magSq(X);
Xp = zeros(size(X,1), size(X,2), size(mask,3));
for s = 1:size(mask,3)
    signal = squeeze(sum(sum(bsxfun(@times, P,   mask(:,:,s)), 1), 2));
    noise  = squeeze(sum(sum(bsxfun(@times, P, 1-mask(:,:,s)), 1), 2));
    snr = signal ./ noise;

    [~,bestChan] = max(snr);
    Xp(:,:,s) = X(:,:,bestChan);
end

% Output spectrogram(s)
Y = Xp .* mask;
