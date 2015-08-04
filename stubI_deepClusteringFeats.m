function [Y data] = stubI_deepClusteringFeats(X, fail, sr, fileName, cleanDir, K, d_m, useKernXcorr, beamformer)

% Extract features for multi-channel deep clustering

if ~exist('silenceThreshold_db', 'var') || isempty(silenceThreshold_db), silenceThreshold_db = -60; end
if ~exist('K', 'var') || isempty(K), K = 15; end
if ~exist('d_m', 'var') || isempty(d_m), d_m = 0.2; end
if ~exist('useKernXcorr', 'var') || isempty(useKernXcorr), useKernXcorr = false; end
if ~exist('beamformer', 'var') || isempty(beamformer), beamformer = 'ds'; end
if ~exist('cleanDir', 'var') || isempty(cleanDir), cleanDir = '/export/ws15-ffs-data2/mmandel/data/merlTest/mix/'; end

wlen = 2*(size(X,1)-1);

% Do clustering and projection
[tdoas itds] = beamSpaceCluster(X, K, sr, d_m, useKernXcorr);
[Xbfn Xbf] = beamSpaceProject(X, tdoas, beamformer);

% Figure out ground truth masks
src1File = strrep(fileName, '.wav', '_src1.wav');
src2File = strrep(fileName, '.wav', '_src2.wav');

[x1 fs] = wavread(fullfile(cleanDir, src1File));
[x2 fs] = wavread(fullfile(cleanDir, src2File));
X1 = stft_multi(x1', wlen);
X2 = stft_multi(x2', wlen);
Xn = X - X1 - X2;
Sil = 0.5*db(mean(magSq(X),3)) > silenceThreshold_db;

loudest = argmax(cat(3, mean(magSq(X1),3), mean(magSq(X2),3), mean(magSq(Xn),3)), 3);
for i = 1:max(loudest(:))
    mask(:,:,i) = loudest == i;
end
mask = bsxfun(@times, mask, ~Sil);
mask = cat(3, mask, Sil);

% Fields for Daniel
data.mag_spec = abs(X(:,:,1));
data.beams = abs(Xbfn);
data.masks = mask;
data.utt_name = basename(fileName);

% Extra bookkeeping fields
data.params.silenceThreshold_db = silenceThreshold_db;
data.params.K = K;
data.params.d_m = d_m;
data.params.useKernXcorr = useKernXcorr;
data.params.beamformer = beamformer;
data.params.sr = sr;
data.params.cleanDir = cleanDir;
