addpath('../mimlib');
addpath('../messl');
addpath('../utils');
workDir = '/home/data/CHiME3/data/audio/16kHz/isolated/';
outDir = '/scratch/near/combination2/';
lstm_dir = '/scratch/near/test_prediction/';
%outDir = '/home/data/CHiME3/data/audio/16kHz/local/clean-spect/'; %temporary

%first create messl masks in local directory





%then combine the masks with lstm mask
enhance_wrapper(@(X, fail, fs, file) stubI_LSTMMessl2(X, fail, fs, file, ...
						      '/scratch/mim/chime3/messlMcMvdrMrf.2Hard5Lbp4Slate/data/', '', 'souden', 'ipd', 'mask', 1,'','',lstm_dir,'average'), ...
workDir, outDir, [1 1], 0, 1, 1,'.dt05.*simu.*\.CH1\.wav',0); 
