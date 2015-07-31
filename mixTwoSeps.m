function mixTwoSeps(inDirA, inDirB, outDir, gainA, gainB, overwrite, chans, files)

if ~exist('chans', 'var') || isempty(chans), chans = ':'; end
if ~exist('overwrite', 'var') || isempty(overwrite), overwrite = ...
        false; end
if ~exist('files', 'var'), files = []; end

if isempty(files)
    filesA = findFiles(inDirA, '.*\.wav$');
    filesB = findFiles(inDirB, '.*\.wav$');
    files = intersect(filesA, filesB);
end

for f = 1:length(files)
    inFileA = fullfile(inDirA, files{f});
    inFileB = fullfile(inDirB, files{f});
    outFile = fullfile(outDir, files{f});
    fprintf('%d: %s\n', f, files{f});

    if exist(outFile, 'file') && ~overwrite,
        fprintf('\b <-- Skipping\n');
        continue;
    end

    [xa fsa] = wavread(inFileA);
    [xb fsb] = wavread(inFileB);
    assert(fsa == fsb);
    fs = fsa;

    % Find two peaks in ILD histogram corresponding to mask = {min,
    % 1}, normalize so that mask = 1 means ILD 0 between Xa and Xb.
    Xa = stft(xa(:,1)', 1024, 1024, 256);
    Xb = stft(xb(:,1)', 1024, 1024, 256);
    ild = db(Xa ./ Xb);
    [h xh] = hist(ild(:), 1000);
    p = pickPeaks(xh, h, 2);
    xa = xa * 10^(-p(2)/20);

    % Mix and save
    x = (gainA * xa + gainB * xb) / (gainA + gainB);
    ensureDirExists(outFile);
    wavwrite(x(:,chans), fs, outFile);
end
