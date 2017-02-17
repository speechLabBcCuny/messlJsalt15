function plotsSpringer16(toDisk, startAt)

if ~exist('toDisk', 'var') || isempty(toDisk), toDisk = false; end
if ~exist('startAt', 'var') || isempty(startAt), startAt = 0; end

outDir = '~/work/papers/springer16/extraFigs/bwFriendly/';

prt('ToFile', toDisk, 'StartAt', startAt, ...
    'Width', 4, 'Height', 3, 'NumberPlots', 0, ...
    'TargetDir', outDir, ...
    'SaveTicks', 1, 'Resolution', 200)

inDir = '/home/mim/work/talks/sane-2015-10-22/jsaltWavs/';

hop_s = 0.016;
fs = 16000;
maxFreq = 8000;
cmap = easymap('kryw', 510);
cax = [-66 24];

[~,inFiles] = findFiles(inDir, '.*.wav');
for f = 1:length(inFiles)
    outName = basename(inFiles{f}, 0);
    [X fs hop_s] = loadSpecgram(inFiles{f});
    prtSpectrogram(db(X), outName, fs, hop_s, cmap, cax, [1 1 1], maxFreq)
end

[~,inFiles] = findFiles(inDir, 'CH1.wav');
ch0file = strrep(inFiles{1}, 'CH1', 'CH0');
ch2file = strrep(inFiles{1}, 'CH1', 'CH2');
ch3file = strrep(inFiles{1}, 'CH1', 'CH3');
[X0 fs hop_s] = loadSpecgram(ch0file);
[X1 fs hop_s] = loadSpecgram(inFiles{1});
[X2 fs hop_s] = loadSpecgram(ch2file);
[X3 fs hop_s] = loadSpecgram(ch3file);
ild01 = db(X0 ./ X1);
ild21 = db(X2 ./ X1);
ild31 = db(X3 ./ X1);
ipd01 = angle(X0 ./ X1);
ipd21 = angle(X2 ./ X1);
ipd31 = angle(X3 ./ X1);

ildCmap = easymap('bwr', 254);
ildCax  = [-20 20];
ipdCmap = easymap('bcyrmb', 254);
ipdCax  = [-pi pi];

prtSpectrogram(ild01, [basename(inFiles{1},0) '_ild01'], fs, hop_s, ildCmap, ildCax, [1 1 1], maxFreq);
prtSpectrogram(ild21, [basename(inFiles{1},0) '_ild21'], fs, hop_s, ildCmap, ildCax, [1 1 1], maxFreq);
prtSpectrogram(ild31, [basename(inFiles{1},0) '_ild31'], fs, hop_s, ildCmap, ildCax, [1 1 1], maxFreq);
prtSpectrogram(ipd01, [basename(inFiles{1},0) '_ipd01'], fs, hop_s, ipdCmap, ipdCax, [1 1 1], maxFreq);
prtSpectrogram(ipd21, [basename(inFiles{1},0) '_ipd21'], fs, hop_s, ipdCmap, ipdCax, [1 1 1], maxFreq);
prtSpectrogram(ipd31, [basename(inFiles{1},0) '_ipd31'], fs, hop_s, ipdCmap, ipdCax, [1 1 1], maxFreq);



function [X fs hop_s] = loadSpecgram(file, win_s)
if ~exist('win_s', 'var') || isempty(win_s), win_s = 0.032; end

[x fs] = wavReadBetter(file);
x = x(:,1);
nfft = 2 * round(win_s * fs / 2);
hop = round(nfft / 4);
X = stft(x', nfft, nfft, hop);
hop_s = hop / fs;
