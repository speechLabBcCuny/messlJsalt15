function deepClusteringResynthesis(wavDir, maskDir, outDir, beamformer, ...
                                   overwrite, wlen)

% Resynthesize wav files from deep clustering masks
%
% deepClusteringResynthesis(wavDir, maskDir, outDir, beamformer,
% overwrite, wlen)
%
% Output will be a single wav file with one source in each
% channel. Directories should all have the same sub-directory
% structure. Beamformer is a string that specifies the beamformer
% to use to reconstruct the signal.  Assumes that input wav files
% also have all of the channels in a single wav file.

if ~exist('overwrite', 'var') || isempty(overwrite), overwrite = ...
        false; end
if ~exist('beamformer', 'var') || isempty(beamformer), beamformer = ...
        'bestMic'; end
if ~exist('wlen', 'var') || isempty(wlen), wlen = 1024; end

files = findFiles(maskDir, '.*\.mat$');

for f = 1:length(files)
    maskFile = fullfile(maskDir, files{f});
    wavFile = fullfile(wavDir, strrep(files{f}, '.mat', '.wav'));
    outFile = fullfile(outDir, strrep(files{f}, '.mat', '.wav'));

    fprintf('%d: %s\n', f, outFile);
    if exist(outFile, 'file') && ~overwrite
        fprintf('\b  Skipping\n');
        continue;
    end
    if ~exist(wavFile, 'file')
        fprintf('\b  Couldn''t find wav file: "%s"\n', wavFile);
        continue;
    end

    d = load(maskFile);
    mask = permute(d.masks, [2 1 3]);

    [x fs] = wavread(wavFile);
    X = stft_multi(x', wlen);
    
    fail = zeros(1, size(X,3));
    switch beamformer
      case {'bestMic', 'best'}
        Xp = pickChanWithBestSnr(X, mask, fail);
      case 'souden'
        [Xp mvdrMask mask] = mvdrSoudenMulti(X, mask, fail);
      otherwise
        error('Unknown beamformer: %s', beamformer)
    end

    Y = Xp .* mask;
    y = istft_multi(Y, size(x,1))';
    ensureDirExists(outFile);
    wavwrite(y, fs, outFile);
end
