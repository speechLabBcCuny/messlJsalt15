
originalDir = '/home/vietanh/data/CHiME3/data/audio/16kHz/isolated/dt05_bth';
estimateDir = '/home/vietanh/speech_class_16/Parchami_stage1/dt05_str_simu/wavout';
resultDir = '/home/vietanh/speech_class_16/Parchami_stage1/OPS_result/result'
inFilesOrPattern = '.*\.wav$';
inFiles = findFiles(estimateDir, inFilesOrPattern);
OPSfile = zeros(length(inFiles),1);
for i = 1:length(inFiles);
    inFileaddbth = strrep(inFiles{i},'STR','BTH');
    inFileaddch0 = strrep(inFileaddbth,'.wav','.CH0.wav');
    a = fullfile(originalDir,inFileaddch0);
    originalFiles = {a};
    estimateFile = fullfile(estimateDir, inFiles{i});
    options.destDir = '/home/vietanh/speech_class_16/Parchami_stage1/Evaluation_temp/';
    options.segmentationFactor = 1; % increase this integer if you experienced "out of memory" problems
    res = PEASS_ObjectiveMeasure(originalFiles,estimateFile,options);
    OPSfile(i) =res.OPS;
end
save(resultDir,'OPSfile')