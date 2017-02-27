function [Y data] = stubI_recoverFromLstm(X, fail, fs, inFile, beamformer,lstm_dir)

% Load a MESSL mask and TDOA and apply it to the input the same way
% stubI_messlMc would.

if ~exist('beamformer', 'var') || isempty(beamformer), beamformer = 'bestMic'; end



%inFile is the filename for the auido file, so make use of it to find LSTM mask file.
lstm_file = fullfile(lstm_dir, regexprep(inFile, '(\.CH1)?\.wav$', '.mat'));
% six masks need to combine to one mask
LSTM_Mask = load(lstm_file);   %something like this
mask = LSTM_Mask.mask.';

wlen = 2*(size(X,1)-1);
M = sum(~fail);

switch beamformer
    case 'bestMic'
        Xp = pickChanWithBestSnr(X, mask, fail);
    otherwise
        error('Unknown beamformer: %s', beamformer)
end

data.mask = mask;
% Output spectrogram(s)
Y = Xp .* mask;
%subplots({db(Xp(:,:,1)), mask(:,:,1)})
1+1;
