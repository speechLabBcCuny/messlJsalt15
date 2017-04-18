addpath('../mimlib');
addpath('../messl');
addpath('../utils');
workDir = '/home/data/CHiME3/data/audio/16kHz/isolated/';
outDir = '/scratch/near/CHiME3/fixed_model/v3/replayMessl/maskhold11/';
combined_mask_dir = '/scratch/near/CHiME3/v3/maskhold11_lstm_initial_messl_output/data/';

%put combined mask to MESSL
enhance_wrapper(@(X, fail, fs, file) stubI_replayMessl(X, fail, fs, file, ...
						       combined_mask_dir, '', 'souden', '', 'mask', 1,9), ...
		workDir,outDir, [5 5], 1, 1, 1,'[de]t05((?!bth).)*\.CH1\.wav$',1);
