function [Xp mvdrMask mask] = maskDrivenMvdrMulti(X, mask, fail, tdoas)

% Wrapper around maskDrivenMvdr for use in stubI_messlMc
% functions. Outputs twice as many channels as sources, the first
% set are MVDR+mask, the second set are MVDR only.

minNoiseObsPerMic = 3;
minNoiseObs = minNoiseObsPerMic * size(X,3);

Xp = zeros(size(X,1), size(X,2), size(mask,3));
for s = 1:size(mask,3)
    if s > size(tdoas,2)
        % No MVDR, pick a random channel
        Xp(:,:,s) = X(:,:,1);
    else
        mvdrMask(:,:,s) = mungeMaskForMvdr(mask(:,:,s), minNoiseObs);
        Xp(:,:,s) = maskDrivenMvdr(X(:,:,~fail), mvdrMask(:,:,s), tdoas(:,s));
    end
end

% Duplicate signals
Xp   = cat(3, Xp, Xp);
mask = cat(3, mask, ones(size(mask)));
