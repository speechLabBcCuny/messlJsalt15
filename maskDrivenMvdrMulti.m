function [Xp mvdrMask mask] = maskDrivenMvdrMulti(X, mask, fail, tdoas, mvdrOnly)

% Wrapper around maskDrivenMvdr for use in stubI_messlMc
% functions. Outputs twice as many channels as sources, the first
% set are MVDR+mask, the second set are MVDR only.

mvdrMask = mungeMaskForMvdr(mask);
Xp = zeros(size(X,1), size(X,2), size(mask,3));
for s = 1:size(mask,3)-1
    Xp(:,:,s) = maskDrivenMvdr(X(:,:,~fail), mvdrMask(:,:,s), tdoas(:,s));
end
Xp(:,:,end) = X(:,:,1);  % Garbage source

if mvdrOnly
    mask = ones(size(mask));
end
