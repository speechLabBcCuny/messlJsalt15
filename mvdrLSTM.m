function Y = mvdrLSTM(X, Ncov, Scov, regulN, refMic, eyeCoef)

% Perform MVDR beamforming using a mask only (no look direction).  
%
%   Y = mvdrSouden(X, M, Ncov, Scov, regulN, refMic, eyeCoef)
%
% See Souden, Benesty, and Affes (2010). Based on their equation (18) with
% beta = 0.
%
% Inputs:
%   X:       FxTxC tensor of complex input spectrograms for each channel, c
%   Ncov:    estimated noise covariances or empty to estimate using the mask
%   Scov:    estimated target covariances or empty to estimate from the mixture
%   regulN:  regularization for covariance inversion, leave at default
%   refMic:  microphone used as reference for reconstruction
%   eyeCoef: how much of the identity to subtract from estimated mix stats
%            1 implements Souden et al's method, 0 might work better
%
% Outputs:
%   Y: FxT matrix of estimated complex spectrum of target

[F T C] = size(X);
wlen = 2*(F-1);
regulM = 0;
minCor = 1;
beta = 0;

if ~exist('Ncov', 'var'), Ncov = []; end
if ~exist('Scov', 'var'), Scov = []; end
if ~exist('regulN', 'var') || isempty(regulN), regulN = 1e-3; end
if ~exist('refMic', 'var') || isempty(refMic), refMic = 1; end
if ~exist('eyeCoef', 'var') || isempty(eyeCoef), eyeCoef = 1; end

pickMic = zeros(C,1);
pickMic(refMic) = 1 / length(refMic);

X = permute(X, [3 2 1]);  % Now it is CxTxF

% Estimate noise covariance and mix covariance
if isempty(Ncov)
    Ncov = zeros(C, C, F);
    for f = 1:F
        Tcov = covw(X(:,:,f)', 1-M(f,:)');
        Ncov(:,:,f) = 0.5 * (Tcov + Tcov');  % Ensure Hermitian symmetry
    end
end

if ~isempty(Scov)
    % Simulate Mcov from Scov and Ncov
    Mcov = Scov + Ncov;
else
    % Estimate mixture covariance
    Mcov = zeros(C, C, F);
    for f = 1:F
        Tcov = covw(X(:,:,f)', ones(size(M(f,:)')));
        Mcov(:,:,f) = 0.5 * (Tcov + Tcov');  % Ensure Hermitian symmetry
    end
end

% MVDR beamforming
Y  = zeros(F,T);
for f = 1:F,
    RNcov = Ncov(:,:,f) + regulN * diag(diag(Mcov(:,:,f)));
    RMcov = Mcov(:,:,f) + regulM * diag(diag(Mcov(:,:,f)));
    num = (RNcov \ RMcov - eyeCoef*eye(C));
    %lambda = real(trace(num));
    lambda = max(minCor, real(trace(num)));
    den = beta + lambda;
    h = (num * pickMic) / den;
    t(f) = real(trace(num));
    Y(f,:) = h' * X(:,:,f);
end
1+1;