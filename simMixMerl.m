function [y fs] = simMixMerl(noiseGain_db)

if ~exist('noiseGain_db', 'var') || isempty(noiseGain_db), noiseGain_db = -60; end

noiseGain = 10^(noiseGain_db/20);

[x1 fs] = wavread('/home/data/merlTest/clean/f1/050c0101.wav');
[x2 fs] = wavread('/home/data/merlTest/clean/f1/050c0102.wav');

[h1 fs] = wavread('/home/data/merlTest/rir/trimmed/h_angB_far.wav');
[h2 fs] = wavread('/home/data/merlTest/rir/trimmed/h_angA_near.wav');
[h3 fs] = wavread('/home/data/merlTest/rir/diffuse/h_angA_far.wav');

y1 = fftfilt(h1, x1);
y2 = fftfilt(h2, x2);

y = mixSignalsWithOffsets(y1, y2, fs, 2, 6);

y3 = noiseGain * fftfilt(h3, randn(size(y,1),1));
y = y + y3;
