function [tdoas itds] = beamSpaceCluster(Y, K, fs, d_m, useKernXcorr)

% Find K directions of arrival of frames of Y, project Y onto delay-and-sum
% beamformers in those directions.

if ~exist('useKernXcorr', 'var') || isempty(useKernXcorr), useKernXcorr = false; end

[F T Ch] = size(Y);
taug = tauGrid(d_m, fs, 31);
wlen = 2*(F-1);
maxPts = 5000;

channelPairs = nchoosek(1:Ch, 2);
if useKernXcorr
    % Use frequency- and time-varying cross-correlations to try to find
    % regions dominated by each source.
    normXCm = zeros(F, T, size(channelPairs,1));
    normXCc = normXCm;
    kern = tfKernel([0.04 250], fs, wlen);
    for cc = 1:size(channelPairs,1)
        normXCT = normKernXcorr(Y(:,:,channelPairs(cc,:)), taug, wlen, kern);
        [normXCm(:,:,cc) normXCc(:,:,cc)] = max(normXCT,[],3);
        %subplots({normXCm(:,:,cc), taug(normXCc(:,:,cc))}, [-1 1])
    end
    highCoh = find(mean(normXCm > 0.4, 3) > 0.75);
    if length(highCoh) > maxPts
        highCoh = highCoh(sort(randsample(length(highCoh), maxPts)));
    end
    nxc = permute(normXCc, [3 1 2]);
    perPair = taug(nxc(:,highCoh));
else
    % Use per-frame cross-correlations
    tdoas = zeros(length(taug), T, size(channelPairs,1));
    for cc = 1:size(channelPairs,1)
        tdoas(:,:,cc) = tdoaOverTime(Y(:,:,channelPairs(cc,:)), taug);
    end
    [~,perPair] = max(tdoas,[],1);
    perPair = taug(squeeze(perPair).');
end

[perMic,estPerPair,~,failed] = perMicTdoaLs(perPair, channelPairs,[],1);

[~,~,index] = kmedoids(estPerPair(:,~failed),K);
tdoas = perMic(:,index(1:K));
itds  = estPerPair(:,index(1:K));
