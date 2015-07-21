function Ybfn = beamSpaceProject(Y, K, fs, d_m)

% Find K directions of arrival of frames of Y, project Y onto delay-and-sum
% beamformers in those directions.

[F T Ch] = size(Y);
taug = tauGrid(d_m, fs, 31);
wlen = 2*(F-1);

channelPairs = nchoosek(1:Ch, 2);
tdoas = 
for cc = 1:size(channelPairs,1)
    T(:,:,cc) = tdoaOverTime(Y(:,:,channelPairs(cc,:)), taug);
end
[~,perPair] = max(T,[],1);
perPair = taug(squeeze(perPair).');
[perMic estPerPair] = perMicTdoaLs(perPair, channelPairs,[],0);

[~,~,index]=kmedoids(estPerPair,K);

Ybf = zeros(size(Y,1),size(Y,2),K);
for kk=1:K
    for ff = 1:F
        Df = sqrt(1/Ch) * exp(-2*1i*pi*(ff-1)/wlen*perMic(:,index(kk))); % steering vector
        Ybf(ff,:,kk) = (Df'*(squeeze(Y(ff,:,:)).')).';
    end
end

Ybfn = bsxfun(@rdivide, Ybf, sqrt(sum(magSq(Ybf),3)));
