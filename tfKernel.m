function [K, t, f] = tfKernel(bw, sr, nfft, N, hop)

% [K, t, f] = tfKernel(bw, sr, nfft, N, hop)
%
% Create a Gaussian time-frequency kernel for use with a spectrogram
% that has FFT size nfft and hop size hop, both measured in samples.
% The sampling rate of the signal is sr (in Hz).  The kernel will be
% size N(2)xN(1) (measured in seconds and Hz, respectively) and the
% kernel will have bandwidth bw(2)xbw(1) where the time bandwidth is
% measured in seconds, and the frequency bandwidth is measured in Hz.
% Note that in the 2-vectors bw and N, time comes first and frequency
% second.

if ~exist('N', 'var'), N = 6*bw; end
if ~exist('hop', 'var'), hop = nfft/4; end

df = sr / nfft;
dt = hop / sr;

t = evenSpace(-N(1)/2, dt, N(1)/2);
f = evenSpace(-N(2)/2, df, N(2)/2);
%t = dt*(-N(1)/2:N(1)/2);
%f = df*(-N(2)/2:N(2)/2);

[T,F] = meshgrid(t, f);

K = exp(-T.^2/(2*bw(1)^2) - F.^2/(2*bw(2)^2));
K = K ./ sqrt(sum(sum(K.^2)));


function s = evenSpace(low, step, high)
% Like low:step:high but rounds low and high to be divisible by
% step.  Rounds low down and high up.

low = floor(low/step)*step;
high = ceil(high/step)*step;
s = low:step:high;
