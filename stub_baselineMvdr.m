function Y = stub_baselineMvdr(X, N, Ncov, fail, TDOA, fs)

% Stub for use with CHiME3_enhance_wrapper implementing baseline MVDR enhancement.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright 2015 Michael Mandel
%                University of Sheffield (Jon Barker, Ricard Marxer)
%                Inria (Emmanuel Vincent)
%                Mitsubishi Electric Research Labs (Shinji Watanabe)
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

regul=1e-3; % MVDR regularization factor

[nbin,nfram,nchan] = size(X);
wlen = 2*(nbin-1);

% MVDR beamforming
Xspec=permute(mean(abs(X).^2,2),[3 1 2]);
Y=zeros(nbin,nfram);
for f=1:nbin,
    for t=1:nfram,
        Xtf= permute(X(f,t,:),[3 1 2]);
        Df = sqrt(1/nchan)*exp(-2*1i*pi*(f-1)/wlen*fs*TDOA(:,t)); % steering vector
        Rt = Ncov(~fail,~fail,f)+regul*diag(Xspec(~fail,f));
        T  = Df(~fail)'/Rt;
        Y(f,t)=(T*Xtf(~fail))/(T*Df(~fail));
        %Y(f,t)=Df(~fail)'/(Ncov(~fail,~fail,f)+regul*diag(Xspec(~fail,f)))*Xtf(~fail)/(Df(~fail)'/(Ncov(~fail,~fail,f)+regul*diag(Xspec(~fail,f)))*Df(~fail));
    end
end
