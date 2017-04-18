addpath('../mimlib');
addpath('../messl');
addpath('../utils');

%option for combination
combination_option = 'max';


workDir = '/home/data/CHiME3/data/audio/16kHz/isolated/';
outDir = '/scratch/near/CHiME3/fixed_model/0411_v2/max/';
lstm_dir = '/scratch/near/CHiME3/fixed_model/v2/lstm-out-0411:14:06/';

%first create messl masks in local directory
messl_dir = '/scratch/mim/chime3/messlMcMvdrMrf.2Hard5Lbp4Slate/data/';
%messl_dir = '/scratch/near/CHiME3/v3/maskhold11_lstm_initial_messl_output/data/';



%then combine the masks with lstm mask
enhance_wrapper(@(X, fail, fs, file) stubI_LSTMMessl2(X, fail, fs, file, ...
<<<<<<< HEAD
    messl_dir, '', 'souden', '', 'mask', 1,9,'',lstm_dir,combination_option),workDir, outDir, [5 5], 1, 1, 1,'.[de]t05((?!bth).)*\.CH1\.wav',1);
=======
    messl_dir, '', 'souden', '', 'mask', 1,9,'',lstm_dir,combination_option), ...
    workDir, outDir, [5 5], 1, 1, 1,'.[de]t05.*\.CH1\.wav',0);
>>>>>>> f279e252beabd04fdff083edc1f9d143d62dd32f

fprintf('Combining LSTM with MESSL, %s option', combination_option);
