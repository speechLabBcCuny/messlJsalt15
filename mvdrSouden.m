function Y = mvdrSouden(X, M, Ncov, Scov, regulN, refMic, beta)

% Perform MVDR beamforming using a mask only.  See Souden, Benesty,
% and Affes (2010). Based on their equation (18) with beta = 0.

[F T C] = size(X);
wlen = 2*(F-1);
regulM = 0;
minCor = 1;

if ~exist('Ncov', 'var'), Ncov = []; end
if ~exist('Scov', 'var'), Scov = []; end
if ~exist('regulN', 'var') || isempty(regulN), regulN = 1e-3; end
if ~exist('refMic', 'var') || isempty(refMic), refMic = 1; end
if ~exist('beta', 'var') || isempty(beta), beta = 0; end

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
    num = (RNcov \ RMcov - 0*eye(C));
    lambda = real(trace(num));
    %lambda = max(minCor, real(trace(num)));
    den = beta + lambda;
    h = (num * pickMic) / den;
    t(f) = real(trace(num));
    % t(f) = den;
    Y(f,:) = h' * X(:,:,f);
end
1+1;
