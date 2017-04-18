addpath('../mimlib');
addpath('../messl');
addpath('../utils');
workDir = '/home/data/CHiME3/data/audio/16kHz/isolated/';
outDir = '/scratch/near/CHiME3/v4/0319_13:44/replayMessl/lstm/';
combined_mask_dir = '/scratch/near/CHiME3/v4/0319_13:44/step1/lstm/data/';

%put combined mask to MESSL
enhance_wrapper(@(X, fail, fs, file) stubI_replayMessl(X, fail, fs, file, ...
  combined_mask_dir, '', 'souden', '', 'mask', 1), ...
  workDir, ...xs
		outDir, [5 5], 0, 1, 1,'[de]t05((?!bth).)*\.CH1\.wav$',1); 
