addpath('../mimlib');
addpath('../messl');
addpath('../utils');
workDir = '/home/data/CHiME3/data/audio/16kHz/isolated/';
outDir = '/scratch/near/CHiME3/v3/0312/maskhold11_messl_output/';
lstm_dir = '/scratch/near/CHiME3/v2/0310/lstm_output/';



enhance_wrapper(@(X, fail, fs, file) stubI_LSTMMessl3(X, fail, fs, file, 1, 0, 0.12, 1, 'bestMic',lstm_dir,...
'mrfCompatExpSched', [0 0 0 0 0 0 0 0 .2], 'mrfHardCompatExp', 5 , ...
'mrfCompatFile', '~mandelm/data8/messlData/ibmNeighborCountsSimple.mat', 'mrfLbpIter', 4, 'vis', 0), ...
 workDir, outDir, [20 20], 0, 0, 1,'.[de]t05.*\.CH1\.wav',0); 
fprintf('15\n');
