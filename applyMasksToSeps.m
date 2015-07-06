function applyMasksToSeps(mvdrDir, maskDir, outMaskedDir, overwrite)

% Apply masks derived by extractMasksFromSeps to single channel noisy wav files
%
% applyMasksToSeps(mvdrDir, maskDir, outMaskedDir, overwrite)

if ~exist('overwrite', 'var') || isempty(overwrite), overwrite = false; end

wlen = 1024; % STFT window length

files = findFiles(mvdrDir, '.*.wav');

for f = 1:length(files)
    mvdrFile = fullfile(mvdrDir, files{f});
    maskFile = fullfile(maskDir, strrep(files{f}, '.wav', '.mat'));
    outMaskedFile = fullfile(outMaskedDir, files{f});
    fprintf('%d: %s\n', f, outMaskedFile);
    
    if exist(outMaskedFile, 'file') && ~overwrite
        fprintf('\b <-- Skipping\n');
        continue
    end
    
    [xn fsn] = wavread(mvdrFile);
    mask = load(maskFile);
    assert(fsn == mask.fs);
    nsampl = size(xn,1);
    fs = fsn;
    
    Xn = stft_multi(xn.', wlen);
    Y = Xn .* mask.mask;
    y = istft_multi(Y, nsampl).';
    y = y * 0.999/max(abs(y(:)));

    ensureDirExists(outMaskedFile);
    wavwrite(y, fs, outMaskedFile);
end
