function plotsWiley16(toDisk, startAt)

if ~exist('toDisk', 'var') || isempty(toDisk), toDisk = false; end
if ~exist('startAt', 'var') || isempty(startAt), startAt = 0; end

inDir  = '~/work/papers/wiley16/code/mandel/inData/';
outDir = '~/work/papers/wiley16/code/mandel/outData/';

prt('ToFile', toDisk, 'StartAt', startAt, ...
    'Width', 4, 'Height', 3, 'NumberPlots', 0, ...
    'TargetDir', outDir, ...
    'SaveTicks', 1, 'Resolution', 200)

maxFreq = 4000;
xrange_s = [1.5 5.5];
cmap = easymap('bcyr', 255);
cax = [-80 10];
ildCmap = easymap('bwr', 254);
ildCax  = [-20 20];
ipdCmap = easymap('bcyrmb', 254);
ipdCax  = [-pi pi];
% maskCmap = easymap('bwr', 254);
maskCmap = jet(508); maskCmap = maskCmap(end/2:end,:);
maskCax = [0 1];

% Figure 12.1 (a) Spectrograms of two individual sound sources and their mixture in
% reverberation (b) Interchannel level difference of two individual sound sources and their
% mixture in reverberation

inFile = 'r09_0003';
[X(:,:,1) fs hop_s] = loadSpecgram(fullfile(inDir, [inFile '.wav']), [], 1);
[X(:,:,2) fs hop_s] = loadSpecgram(fullfile(inDir, [inFile '.wav']), [], 2);

X = X / 64;  % Fix scaling...

prtSpectrogram(db(X(:,:,1)), [inFile '_ch1'], fs, hop_s, cmap, cax, [1 1 1], maxFreq, xrange_s);
prtSpectrogram(db(X(:,:,2)), [inFile '_ch2'], fs, hop_s, cmap, cax, [1 1 1], maxFreq, xrange_s);

ild = db(X(:,:,1) ./ X(:,:,2));
ipd = angle(X(:,:,1) ./ X(:,:,2));

prtSpectrogram(ild, [inFile '_ild'], fs, hop_s, ildCmap, ildCax, [1 1 1], maxFreq, xrange_s);
prtSpectrogram(ipd, [inFile '_ipd'], fs, hop_s, ipdCmap, ipdCax, [1 1 1], maxFreq, xrange_s);


% Figure 12.3 Histogram of ITD and ILD features used by DUET from two-channel,
% reverberant recording

d_cmps = 6.3;
I = 3;
vis = 0;
peaks = [-.2 5; -.15 1; -.1 -8];
[~,duetMasks,duetHist,duetA,duetD] = duet(X(2:end-1,:,:), fs, d_cmps, I, [], vis, peaks);

subplot(1,1,1)
imagesc(duetA, duetD, duetHist);
axis xy
colormap(cmap);
colorbar
hold on
plot(peaks(:,1), peaks(:,2), 'ow')
hold off
xlabel('ILD (dB)'), ylabel('ITD (samples)')
prt([inFile '_duetHist'])

for i = 1:I
    prtSpectrogram(duetMasks(:,:,i), sprintf('%s_duet%d', inFile, i), fs, hop_s, maskCmap, maskCax, [1 1 1], maxFreq, xrange_s);
end

% Figure 12.4 Masks estimated by several two-channel clustering systems on reverberant
% mixture.

messlData = load(fullfile(outDir, 'messlOutput'));

for i = 1:size(messlData.mask,3)
    prtSpectrogram(squeeze(messlData.p_lr_iwt(1,:,:,i)), sprintf('%s_messlSoft%d', inFile, i), fs, hop_s, maskCmap, maskCax, [1 1 1], maxFreq, xrange_s);
    prtSpectrogram(messlData.mask(:,:,i), sprintf('%s_messlHard%d', inFile, i), fs, hop_s, maskCmap, maskCax, [1 1 1], maxFreq, xrange_s);
end



% Figure 12.8 Masks estimated by several two-channel classification systems on
% reverberant mixture.
