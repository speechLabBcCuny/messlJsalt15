function mixTwoSeps(inDirA, inDirB, outDir, gainA, gainB, overwrite, chans, files, renameFn)

if ~exist('inDirB', 'var'), inDirB = ''; end
if ~exist('chans', 'var') || isempty(chans), chans = ':'; end
if ~exist('overwrite', 'var') || isempty(overwrite), overwrite = ...
        false; end
if ~exist('files', 'var'), files = []; end
if ~exist('renameFn', 'var') || isempty(renameFn), renameFn = @(x) x; end

bSigInExtraChans = isempty(inDirB);

if isempty(files)
    filesA = findFiles(inDirA, '.*\.wav$');
    if ~bSigInExtraChans
        filesB = findFiles(inDirB, '.*\.wav$');
        files = intersect(filesA, filesB);
    else
        files = filesA;
    end
end

for f = 1:length(files)
    outFile = fullfile(outDir, renameFn(files{f}));
    fprintf('%d: %s\n', f, renameFn(files{f}));

    if exist(outFile, 'file') && ~overwrite,
        fprintf('\b <-- Skipping\n');
        continue;
    end

    inFileA = fullfile(inDirA, files{f});
    [xa fsa] = wavread(inFileA);

    if bSigInExtraChans
        xb = xa(:,end/2+1:end);
        xa = xa(:,1:end/2);
        fs = fsa;
    else
        inFileB = fullfile(inDirB, files{f});
        [xb fsb] = wavread(inFileB);
        assert(fsa == fsb);
        fs = fsa;
    end

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
