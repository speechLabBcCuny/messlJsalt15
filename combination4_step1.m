addpath('../mimlib');
addpath('../messl');
addpath('../utils');

%option for combination
combination_option = 'max';


workDir = '/home/data/CHiME3/data/audio/16kHz/isolated/';
outDir = '/scratch/near/CHiME3/fixed_model/v4/max/';
lstm_dir = '/scratch/near/CHiME3/v4/lstm_output_max_0406:12:03/';

%first create messl masks in local directory
messl_dir = '/scratch/mim/chime3/messlMcMvdrMrf.2Hard5Lbp4Slate/data/';




%then combine the masks with lstm mask
enhance_wrapper(@(X, fail, fs, file) stubI_LSTMMessl2(X, fail, fs, file, ...
    messl_dir, '', 'souden', '', 'mask', 1,9,'',lstm_dir,combination_option), ...
    workDir, outDir, [5 5], 1, 1, 1,'[de]t05((?!bth).)*\.CH1\.wav$',1); 
fprintf('Combining LSTM with MESSL, average option');

