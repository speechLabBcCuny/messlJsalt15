function Y = stub_messlMcIld2(X, N, Ncov, fail, TDOA, fs)

% Multichannel MESSL mask applied to MVDR output initialized from ILD
% between MVDR output and mic 2 (rear-facing)

threshold_db = 12;
maxSup_db = -40;

maxSup = 10^(maxSup_db/20);

c    = 340;   % speed of sound in air [M/s]
d    = 0.35;  % slightly more than the acoustic distance between microphones [M]
nTau = 31;    % number of tau samples to use
maxItd = 2 * fs * d / c;  % maximum ITD [samples]
tau = linspace(-maxItd, maxItd, nTau);

% MVDR for linear beamforming
mvdr = stub_baselineMvdr(X, N, Ncov, fail, TDOA, fs);

if ~fail(2)
    % Compute ILD between MVDR output and mic 2 (rear-facing)
    maskInit = maxSup + (1-maxSup)*(db(magSq(mvdr(2:end-1,:)) ./ magSq(X(2:end-1,:,2))) > threshold_db);
    maskInit = cat(3, maskInit, 1 - maskInit);
else
    maskInit = [];
end

% MESSL for mask
messlOpts = {'GarbageSrc', 1, 'fixIPriors', 1, 'mrfCompatExpSched', [0 0 0 0 0 0 0 0 .1], ...
    'mrfCompatFile', '~mandelm/data8/messlData/ibmNeighborCountsSimple.mat', ...
    'maskInit', maskInit};
[p_lr_iwt params hardMasks] = messlMultichannel(X(2:end-1,:,~fail), tau, 1, messlOpts{:});

z = zeros(1, size(X,2));
mask = [z; squeeze(p_lr_iwt(1,:,:,1)); z];
mask = maxSup + (1 - maxSup) * mask;

% Output spectrogram
Y = mvdr .* mask;
