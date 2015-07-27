function Ybf = mvdrHendriksLoop(Y, tdoas, F, Ch, wlen)
% Use formulation of MVDR from Hendriks & Gerkmann (2012):
% w = inv(Rxx)*d / (d'*inv(Rxx)*x)
% Where Rxx is the estimated spatial covariance of the noise
alpha = 0.03;
Ybf = zeros(size(Y,3), size(Y,2));
for ff = 1:F
    Df = sqrt(1/Ch) * exp(2*1i*pi*(ff-1)/wlen*tdoas); % steering vector

    dnm = Df * (1./Df).';
    lastETilde = eye(Ch);
    for tt = 1:size(Y,2)
        eTildeUpdate = Y(:,tt,ff) * Y(:,tt,ff)' - dnm * diag(magSq(Y(:,tt,ff)));
        eTilde = (1-alpha)*lastETilde + alpha * eTildeUpdate;
        lastETilde = eTilde;
        eHat = 0.5 * (eTilde + eTilde');

        invRpyDf = eHat \ Df;
        w = invRpyDf / (Df' * invRpyDf);
        Ybf(ff,tt) = w'*Y(:,tt,ff);
    end
    
%     Ryy = cov(Y(:,:,ff).');
%     Rpy = Ryy - Df * (diag(Ryy)./Df).';
%     A = Rpy;
%     
%     invADf = A \ Df;
%     w = invADf / (Df' * invADf);
%     Ybf(ff,:) = w'*Y(:,:,ff);
end
