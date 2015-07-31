function [Ybfn Yn] = beamSpaceProject(Y, tdoas, beamformer)

% Find K directions of arrival of frames of Y, project Y onto delay-and-sum
% beamformers in those directions.

if ~exist('beamformer', 'var') || isempty(beamformer), beamformer = 'delayAndSum'; end

[F T Ch] = size(Y);
K = size(tdoas, 2);
wlen = 2*(F-1);

Ybf = zeros(size(Y,1),size(Y,2),K);
Y = permute(Y, [3 2 1]);  % Chan x Time x Freq
for kk=1:K
    switch beamformer
        case {'delayAndSum', 'ds', 'dns'}
            Ybf(:,:,kk) = delayAndSum(Y, tdoas(:,kk), F, Ch, wlen);
        case {'mvdr', 'mvdrHendriks', 'mvdrHendriksVectorized'}
            % Not great
            Ybf(:,:,kk) = mvdrHendriksVectorized(Y, tdoas(:,kk), F, Ch, wlen);
        case 'mvdrHendriksLoop'  
            % Slow, not great
            Ybf(:,:,kk) = mvdrHendriksLoop(Y, tdoas(:,kk), F, Ch, wlen);
        case 'mvdrLcmv' 
            % Inaccurate
            Ybf(:,:,kk) = mvdrLcmv(Y, tdoas(:,kk), F, Ch, wlen);
        otherwise
            error('Unknown beamformer: %s', beamformer);
    end
end

Yn = sqrt(sum(magSq(Ybf),3));
Ybfn = bsxfun(@rdivide, Ybf, Yn);


function Ybf = delayAndSum(Y, tdoas, F, Ch, wlen)   
Ybf = zeros(size(Y,3), size(Y,2));
for ff = 1:F
    Df = sqrt(1/Ch) * exp(2*1i*pi*(ff-1)/wlen*tdoas); % steering vector
    Ybf(ff,:) = Df'*Y(:,:,ff);
end

function Ybf = mvdrLcmv(Y, tdoas, F, Ch, wlen)
% Use formulation of MVDR from LCMV (Van Veen & Buckley, 2000, ch61):
% min_w w'*Ryy*w subject to C'*w = f
% => w = inv(Ryy)*C*inv(C'*inv(Ryy)*C)*f
% Where Ryy is the spatial covariance of the mixture.
% For MVDR, C is just the steering vector and f = 1.
Ybf = zeros(size(Y,3), size(Y,2));
for ff = 1:F
    Ryy = cov(Y(:,:,ff).');
    Ryy = 0.5 * (Ryy + Ryy');  % Ensure Hermitian
    
    Df = sqrt(1/Ch) * exp(2*1i*pi*(ff-1)/wlen*tdoas); % steering vector
    invRyyDf = Ryy \ Df;
    w = invRyyDf / (Df' * invRyyDf);  % * 1;
    Ybf(ff,:) = w'*Y(:,:,ff);
end
