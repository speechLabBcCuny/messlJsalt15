function [Xp mvdrMask mask] = mvdrSoudenMulti(X, mask, fail, Ncov, Scov, eyeCoef)

% Wrapper around mvdrSouden for use in stubI_messlMc
% functions. Outputs twice as many channels as sources, the first
% set are MVDR+mask, the second set are MVDR only.

if ~exist('Ncov', 'var'), Ncov = []; end
if ~exist('Scov', 'var'), Scov = []; end
if ~exist('eyeCoef', 'var'), eyeCoef = []; end
if ~isempty(Ncov), Ncov = Ncov(~fail,~fail,:); end
if ~isempty(Scov), Scov = Scov(~fail,~fail,:); end

minNoiseObsPerMic = 3;
minNoiseObs = minNoiseObsPerMic * size(X,3);

Xp = zeros(size(X,1), size(X,2), size(mask,3));
mvdrMask = zeros(size(mask));
for s = 1:size(mask,3)
    mvdrMask(:,:,s) = mungeMaskForMvdr(mask(:,:,s), minNoiseObs);
    Xp(:,:,s) = mvdrSouden(X(:,:,~fail), mvdrMask(:,:,s), Ncov, Scov, [], [], eyeCoef);
end

% Duplicate signals
Xp   = cat(3, Xp, Xp);
mask = cat(3, mask, ones(size(mask)));
