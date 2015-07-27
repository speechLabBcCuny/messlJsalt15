function Ybf = mvdrHendriksVectorized(Y, tdoas, F, Ch, wlen)
% Use formulation of MVDR from Hendriks & Gerkmann (2012):
% w = inv(Rxx)*d / (d'*inv(Rxx)*x)
% Where Rxx is the estimated spatial covariance of the noise
alpha = 0.03;
Ybf = zeros(size(Y,3), size(Y,2));
for ff = 1:F
    Df = sqrt(1/Ch) * exp(2*1i*pi*(ff-1)/wlen*tdoas); % steering vector

    Ryy = 1/size(Y,2) * Y(:,:,ff) * Y(:,:,ff)';
    Ry2 = mean(magSq(Y(:,:,ff)), 2);
    eTilde = Ryy - Df * (Ry2./Df).';
    eHat = 0.5 * conj(eTilde + eTilde');
    % eHat = conj(eTilde);
    
    invRpyDf = eHat \ Df;
    w = invRpyDf / (Df' * invRpyDf);
    Ybf(ff,:) = w'*Y(:,:,ff);
end
