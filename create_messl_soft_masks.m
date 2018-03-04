function a = create_messl_soft_masks(num_job,num_CPU)
%script for creating all the CHiME3 masks

addpath('../mimlib');
addpath('../messl');
addpath('../utils');


workDir = '/home/data/CHiME3/data/audio/16kHz/isolated/';
outDir = '/scratch/near/CHiME3/MESSL_softmasks/';


%num_CPU = 10
%parfor i = 1+10*(num_job-1):num_job*10
enhance_wrapper(@(X, fail, fs, file) stubI_messlMc(X, fail, fs, file, 1, 0, 0.12, 0, 'bestMic', ...
  'mrfCompatExpSched', [0 0 0 0 0 0 0 0 .2], 'mrfHardCompatExp', 5 , ...
  'mrfCompatFile', 'messl/ibmNeighborCountsSimple.mat', 'mrfLbpIter', 4), ...
  workDir, ...
  outDir, [num_job num_CPU], 0, 0, 1);
%end
