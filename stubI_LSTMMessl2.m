function [Y data] = stubI_LSTMMessl2(X, fail, fs, inFile, loadDataDir, statsDir, beamformer, tdoaSrc, ncovSrc, I, maxSup_db, d_m,lstm_dir, combineOpt)

% Load a MESSL mask and TDOA and apply it to the input the same way
% stubI_messlMc would.

if ~exist('statsDir', 'var'), statsDir = ''; end
if ~exist('beamformer', 'var') || isempty(beamformer), beamformer = 'bestMic'; end
if ~exist('tdoaSrc', 'var') || isempty(tdoaSrc), tdoaSrc = 'dataItd'; end
if ~exist('ncovSrc', 'var') || isempty(ncovSrc), ncovSrc = 'mask'; end
if ~exist('I', 'var') || isempty(I), I = inf; end
if ~exist('maxSup_db', 'var') || isempty(maxSup_db), maxSup_db = 40; end
if ~exist('d_m', 'var') || isempty(d_m), d_m = 0.12; end
if ~exist('combineOpt') || isempty(combineOpt), combineOpt = 'average'; end


%inFile is the filename for the auido file, so make use of it to find LSTM mask file.
lstm_file = fullfile(lstm_dir, regexprep(inFile, '(\.CH1)?\.wav$', '.mat'));
% six masks need to combine to one mask
LSTM_Mask = load(lstm_file);   %something like this
LSTM_Mask = LSTM_Mask.mask.';

wlen = 2*(size(X,1)-1);
M = sum(~fail);

% Load MESSL data structure
refFile = fullfile(loadDataDir, regexprep(inFile, '(\.CH1)?\.wav$', '.mat'));
d = load(refFile);

if strcmp(tdoaSrc, 'ipd') || strcmp(ncovSrc, 'ipd')
tau = tauGrid(d_m, fs, 31);          % Brittle
    covs = covsFromIpdParams(X, d.data.params.ipdParams, M, tau, fs);
    assert(~any(isnan(covs(:))))
end

Scov = [];
switch tdoaSrc
    case 'recomputeItd'
        channelPairs = nchoosek(1:sum(~fail), 2);  % Brittle
        tau = tauGrid(d_m, fs, 31);          % Brittle
        pTauI = cat(3, d.data.params.ipdParams.p_tauI);
        perPairItd = squeeze(sum(bsxfun(@times, tau, pTauI), 2) ./ sum(pTauI,2))';  % posterior mean
        perMicTdoa = perMicTdoaLs(perPairItd(:,size(d.data.params.perMicTdoa,2)), channelPairs);
    case 'dataItd'
        perMicTdoa = d.data.params.perMicTdoa;
    case 'ipd'
        Scov = covs(:,:,:,1);
    otherwise
        error('Unknown tdoa method: %s', tdoaSrc)
end

Ncov = [];
switch ncovSrc
    case 'file'
        % Load precomputed stats for noise covariance.
        % Forces ignoring mask for noise covariance estimation
        statsFile = fullfile(statsDir, regexprep(inFile, '(\.CH1)?\.wav', '.mat'));
        if exist(statsFile, 'file')
            stats = load(statsFile);
            Ncov = stats.Ncov;
        else
            fprintf('No stats found for: "%s"\n', statsFile);
        end
    case 'mask'
        % Ncov = [];
    case 'ipd'
        Ncov = covs(:,:,:,2);
    otherwise
        error('Unknown ncovSrc: %s', ncovSrc)
end

%Mix the MESSL mask with LSTM mask
%Or there could be multi options for the mixture, such as average, max, min 
mask = d.data.mask(:,:,1:min(I,end));
switch combineOpt                                                                                                                     
    case 'average'                                                                                                                    
        mask = (LSTM_Mask+mask)/2;                                                                                                   
    case 'max'                                                                                                                        
        mask = max(LSTM_Mask,mask);                                                                                                   
    case 'mim'                                                                                                                        
        mask = min(LSTM_Mask,mask);
    case 'messl'
        mask = mask;
    case 'lstm'
        mask = LSTM_Mask;
end;
data.origDataFile = refFile;
switch beamformer
    case 'mic1'
        Xp = repmat(X(:,:,1), [1 1 size(mask,3)]);
    case 'mic2'
        Xp = repmat(X(:,:,2), [1 1 size(mask,3)]);
    case 'bestMic'
        Xp = pickChanWithBestSnr(X, mask, fail);
    case 'mvdr'
        [Xp mvdrMask mask] = maskDrivenMvdrMulti(X, mask, fail, perMicTdoa, Ncov);
        data.mvdrMask2 = mvdrMask;
    case 'souden'
        [Xp mvdrMask mask] = mvdrSoudenMulti(X, mask, fail, Ncov, Scov);
        data.mvdrMask2 = single(mvdrMask);
    case 'souden0'
        [Xp mvdrMask mask] = mvdrSoudenMulti(X, mask, fail, Ncov, Scov, 0);
        data.mvdrMask2 = single(mvdrMask);
    otherwise
        error('Unknown beamformer: %s', beamformer)
end
maxSup = 10^(-maxSup_db / 20);
mask = max(mask, maxSup);




data.mask = mask;
data.params = d.data.params;
% Output spectrogram(s)
Y = Xp .* mask;
%subplots({db(Xp(:,:,1)), mask(:,:,1)})
1+1;
