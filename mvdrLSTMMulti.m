function [Xp mvdrMask mask] = mvdrLSTMMulti(X, fail, speechMask, noiseMask,  eyeCoef)

% Wrapper around mvdrSouden for use in stubI_messlMc
% functions. Outputs twice as many channels as sources, the first
% set are MVDR+mask, the second set are MVDR only.

if ~exist('Ncov', 'var'), Ncov = []; end
if ~exist('Scov', 'var'), Scov = []; end
if ~exist('eyeCoef', 'var'), eyeCoef = []; end


minNoiseObsPerMic = 3;
minNoiseObs = minNoiseObsPerMic * size(X,3);

[F T C] = size(X);
wlen = 2*(F-1);

X = permute(X, [3 2 1]);  % Now it is CxTxF

% Estimate noise covariance
Ncov = zeros(C, C, F);
for f = 1:F
    %Ncov(:,:,f) = X(:,:,f) * bsxfun(@times, X(:,:,f), 1-M(f,:))';
    Tcov = covw(X(:,:,f)', 1-noiseMask(f,:)');
    Ncov(:,:,f) = 0.5 * (Tcov + Tcov');  % Ensure Hermitian symmetry
end


%Estimate speech covariance
Scov = zeros(C, C, F);
for f = 1:F
    %Ncov(:,:,f) = X(:,:,f) * bsxfun(@times, X(:,:,f), 1-M(f,:))';
    Tcov = covw(X(:,:,f)', speechMask(f,:)');
    Scov(:,:,f) = 0.5 * (Tcov + Tcov');  % Ensure Hermitian symmetry
end

%exclude the failure channel
Ncov = Ncov(~fail,~fail,:);
Scov = Scov(~fail,~fail,:);
X = permute(X,[3,2,1]); % Now X is FxTxC

Xp = zeros(size(X,1), size(X,2), size(mask,3));
mvdrMask = zeros(size(mask));
for s = 1:size(mask,3)
    mvdrMask(:,:,s) = mungeMaskForMvdr(mask(:,:,s), minNoiseObs);
    Xp(:,:,s) = mvdrLSTM(X(:,:,~fail), mvdrMask(:,:,s), Ncov, Scov, [], [], eyeCoef);
end

% % Duplicate signals
%Xp   = cat(3, Xp, Xp);
%mask = cat(3, mask, ones(size(mask)));
