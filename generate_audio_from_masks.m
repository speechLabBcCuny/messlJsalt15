function a = generate_audio_from_masks(combination_option, lstm_mask_dir, audio_dir,part,parts)

addpath('../mimlib');
addpath('../messl');
addpath('../utils');


workDir = '/home/data/CHiME3/data/audio/16kHz/isolated/';
%outDir = '/scratch/near/CHiME3/fixed_model/0412_v2/lstm/';
%lstm_dir = '/scratch/near/CHiME3/fixed_model/v2/lstm-out-0412:15:42/';

%first create messl masks in local directory
messl_dir = '/scratch/mim/chime3/messlMcMvdrMrf.2Hard5Lbp4Slate/data/';
%messl_dir = '/scratch/near/CHiME3/v3/maskhold11_lstm_initial_messl_output/data/';



%then combine the masks with lstm mask
enhance_wrapper(@(X, fail, fs, file) stubI_LSTMMessl2(X, fail, fs, file, ...
    messl_dir, '', 'souden', '', 'mask', 1,9,'',lstm_mask_dir,combination_option),workDir, audio_dir, [part, parts], 1, 1, 1,'.[de]t05((?!bth).)*\.CH1\.wav',1);
fprintf('Combining LSTM with MESSL, %s option', combination_option);
