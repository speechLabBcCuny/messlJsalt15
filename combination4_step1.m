addpath('../mimlib');
addpath('../messl');
addpath('../utils');

%option for combination
combination_option = 'lstm';


workDir = '/home/data/CHiME3/data/audio/16kHz/isolated/';
outDir = '/scratch/near/CHiME3/v4/';
lstm_dir = '/scratch/near/CHiME3/lstm_output_c4/';

%first create messl masks in local directory
messl_dir = '/scratch/mim/chime3/messlMcMvdrMrf.2Hard5Lbp4Slate/data/';




%then combine the masks with lstm mask
enhance_wrapper(@(X, fail, fs, file) stubI_LSTMMessl2(X, fail, fs, file, ...
    messl_dir, '', 'souden', 'ipd', 'mask', 1,'','',lstm_dir,combination_option), ...
    workDir, outDir, [1 5], 0, 1, 1,'[de]t05((?!bth).)*\.CH1\.wav$',0); 
fprintf('Combining LSTM with MESSL, average option');

