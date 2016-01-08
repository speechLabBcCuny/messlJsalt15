function plotMultichanIldIpdStats(wavFile_or_X)

% Plot interaural level and phase difference stats for all pairs of mics

wlen = 1024;

if isstr(wavFile_or_X)
    wavFile = wavFile_or_X;
    [x fs] = wavread(wavFile);
    X = stft_multi(x.', wlen);
else
    X = wavFile_or_X;
end
    
%tau = tauGrid(0.35, fs, 31);
tau = 0;

Ch = size(X, 3);
channelPairs = nchoosek(1:Ch, 2);
Np = size(channelPairs,1);

for c = 1:Np
    cp = channelPairs(c,:);
    [A{c} angE{c}] = messlObsDerive(X(:,:,cp), tau, wlen);
    names{c} = sprintf('%d-%d', cp);
end
figure(1)
subplots(A, [1 -1], names, @(r,c,i) caxis([-20 20]))
figure(2)
subplots(angE, [1 -1], names, @(r,c,i) colormap(hsv(254)))

for c = 1:Ch
    chNames{c} = sprintf('%d', c);
end

figure(3)
subplots(cellFrom3D(db(X)), [1 -1], chNames, @(r,c,i) caxis([-100 0]))
