function Ybfn = beamSpaceProject(Y, K, fs, d_m)

% Find K directions of arrival of frames of Y, project Y onto delay-and-sum
% beamformers in those directions.

[F T Ch] = size(Y);
taug = tauGrid(d_m, fs, 31);
wlen = 2*(F-1);

channelPairs = nchoosek(1:Ch, 2);
tdoas = zeros(length(taug), T, size(channelPairs,1));
for cc = 1:size(channelPairs,1)
    tdoas(:,:,cc) = tdoaOverTime(Y(:,:,channelPairs(cc,:)), taug);
end
[~,perPair] = max(tdoas,[],1);
perPair = taug(squeeze(perPair).');
[perMic estPerPair] = perMicTdoaLs(perPair, channelPairs,[],0);

[~,~,index] = kmedoids(estPerPair,K);

Ybf = zeros(size(Y,1),size(Y,2),K);
for kk=1:K
    Ybf(:,:,kk) = delayAndSum(Y, perMic(:,index(kk)), F, Ch, wlen);
end

Ybfn = bsxfun(@rdivide, Ybf, sqrt(sum(magSq(Ybf),3)));



function Ybf = delayAndSum(Y, tdoas, F, Ch, wlen)

Ybf = zeros(size(Y,1), size(Y,2));
for ff = 1:F
    Df = sqrt(1/Ch) * exp(2*1i*pi*(ff-1)/wlen*tdoas); % steering vector
    Ybf(ff,:) = (Df'*(squeeze(Y(ff,:,:)).')).';
end
