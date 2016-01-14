function [Y data] = stubI_supervisedMvdr(X, fail, fs, inFile, maskQuantile, adaptQuantile, maxSup_db, includeRef)

% Load a MESSL mask and TDOA and apply it to the input the same way
% stubI_messlMc would.

if ~exist('maskQuantile', 'var') || isempty(maskQuantile), maskQuantile = 0.6; end
if ~exist('adaptQuantile', 'var') || isempty(adaptQuantile), adaptQuantile = 0.8; end
if ~exist('maxSup_db', 'var') || isempty(maxSup_db), maxSup_db = -40; end
if ~exist('includeRef', 'var') || isempty(includeRef), includeRef = false; end

Scov = [];
Ncov = [];

Xdb = db(X);

mask = bsxfun(@gt, Xdb(:,:,1), quantile(Xdb(:,:,1), maskQuantile, 2));
adapt = bsxfun(@gt, Xdb(:,:,1), quantile(Xdb(:,:,1), adaptQuantile, 2));

if ~includeRef
    X = X(:,:,2:end);
    fail = fail(2:end);
end
    
[Xp mvdrMask adapt] = mvdrSoudenMulti(X, adapt, fail, Ncov, Scov);
data.mvdrMask = single(mvdrMask);

mask = cat(3, mask, ones(size(mask)));

maxSup = 10^(-maxSup_db / 20);
mask = max(mask, maxSup);
data.mask = mask;

% Output spectrogram(s)
Y = Xp .* mask;
subplots({lim(db(Xp(:,:,1)), -100, 0), mask(:,:,1), adapt(:,:,1)}, [1 -1])
drawnow
1+1;

