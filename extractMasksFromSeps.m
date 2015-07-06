function extractMasksFromSeps(mvdrDir, maskedWavDir, outMaskDir, overwrite)

% Derive masks from single channel noisy and masked wav files
%
% extractMasksFromSeps(mvdrDir, maskedWavDir, outMaskDir, overwrite)
%
% For files that were generated without saving the masks, re-derive what
% they must have been.  This is slightly trickier because the files might
% have been normalized in amplitude as well before saving.  So re-scale all
% masks to have a maximum value of 1, and whatever minimum relative to
% that.

if ~exist('overwrite', 'var') || isempty(overwrite), overwrite = false; end

wlen = 1024; % STFT window length

files = findFiles(mvdrDir, '.*.wav');

for f = 1:length(files)
    mvdrFile = fullfile(mvdrDir, files{f});
    maskedWavFile = fullfile(maskedWavDir, files{f});
    outMaskFile = fullfile(outMaskDir, strrep(files{f}, '.wav', '.mat'));
    fprintf('%d: %s\n', f, outMaskFile);
    
    if exist(outMaskFile, 'file') && ~overwrite
        fprintf('\b <-- Skipping\n');
        continue
    end
    
    [xn fsn] = wavread(mvdrFile);
    [xm fsm] = wavread(maskedWavFile);
    assert(fsn == fsm);
    assert(all(size(xn) == size(xm)));
    nsampl = size(xn,1);
    fs = fsn;
    
    Xn = stft_multi(xn.', wlen);
    Xm = stft_multi(xm.', wlen);

    mask = abs(Xm) ./ abs(Xn);
    
    [h x] = hist(mask(:), ls10(1e-3, 1e3, 1000));
    logPeaks = pickPeaks(log10(x), h, 2);
    peaks = 10.^(logPeaks);
    % plot(log10(x), h)
    
    mask = single(min(1, mask ./ peaks(end)));
    ensureDirExists(outMaskFile);
    save(outMaskFile, 'mask', 'wlen', 'fs')
end