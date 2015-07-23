function Y = maskDrivenMvdr(X, M, tdoa, fail)

% Perform MVDR beamforming using a mask and a look direction
%
% Inputs
%   X      FxTxC array of complex spectrograms for each mic
%   M      FxT matrix of a mask for all channels where 1 is target, 0 not
%   tdoa   Cx1 vector of time delays at each microphone measured in samples
%   fail   Cx1 binary vector indicating whether each mic has failed
%
% Outputs
%   Y      FxT matrix of a complex spectrogram of the MVDR output

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright 2015 Michael Mandel
%                University of Sheffield (Jon Barker, Ricard Marxer)
%                Inria (Emmanuel Vincent)
%                Mitsubishi Electric Research Labs (Shinji Watanabe)
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

regul=1e-3; % MVDR regularization factor

[F T C] = size(X);
wlen = 2*(F-1);

X = permute(X, [3 2 1]);  % Now it is CxTxF
X = X(~fail,:,:);
tdoa = tdoa(~fail);

% Estimate noise covariance
Ncov = zeros(C, C, F);
for f = 1:F
    %Ncov(:,:,f) = X(:,:,f) * bsxfun(@times, X(:,:,f), 1-M(f,:))';
    Tcov = covw(X(:,:,f)', 1-M(f,:)');
    Ncov(:,:,f) = 0.5 * (Tcov + Tcov');  % Ensure Hermitian symmetry
end

% MVDR beamforming
Xa = squeeze(mean(abs(X).^2,2));
Y  = zeros(F,T);
for f = 1:F,
    Df  = sqrt(1/C) * exp(2*1i*pi*(f-1)/wlen*tdoa);  % steering vector
    Dfs(:,f) = Df;
    Rt  = Ncov(:,:,f) + regul * diag(Xa(:,f));            % regularized noise covariance
    DR  = Df'/Rt;
    Y(f,:) = (DR * X(:,:,f)) / real(DR * Df);
end

% plot(angle(Dfs)')