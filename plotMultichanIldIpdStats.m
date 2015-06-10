function plotMultichanIldIpdStats(wavFile)

% Plot interaural level and phase difference stats for all pairs of mics

wlen = 1024;

[x fs] = wavread(wavFile);
X = stft_multi(x.', wlen);
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
subplots(A, [], names, @(r,c,i) caxis([-20 20]))
figure(2)
subplots(angE, [], names, @(r,c,i) colormap(hsv(254)))
