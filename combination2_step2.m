addpath('../mimlib');
addpath('../messl');
addpath('../utils');
workDir = '/home/data/CHiME3/data/audio/16kHz/isolated/';
outDir = '/scratch/near/replayMessl/';
lstm_dir = '/scratch/near/test_prediction/';

%first create messl masks in local directory
%put combined mask to MESSL
enhance_wrapper(@(X, fail, fs, file) stubI_replayMessl(X, fail, fs, file, ...
  '/scratch/near/combination2/data/', '', 'souden', '', 'mask', 1), ...
  workDir, ...
		outDir, [1 1], 0, 1, 1,'dt05.*simu.*\.CH1\.wav$'); 
