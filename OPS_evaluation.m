function OPS_evaluation(originalDir,estimateDir,method,environment)
% originalDir: Groundtrue Directory, which contain wav file
% estimateDir: files that need to evaluate. 
% method: method folder name, for example: 'Souden', 'LSTM',....
% environment: 'CAF','BUS'.....
inpath_split = strsplit(estimateDir,'/');
estimateDir1= strcat((strrep(estimateDir,char(inpath_split(end)),'')),method);
 % for example: turn a/b/c/wav to a/b/c/method 
mkdir(estimateDir1,'Evaluation/OPS_result')
mkdir(estimateDir1,'Evaluation/Temp')
resultDir = fullfile(estimateDir1,'Evaluation/OPS_result/score')
tempDir = fullfile(estimateDir1,'Evaluation/Temp/') % Please add / after Temp if not it will save at Evaluation folder
inFilesOrPattern = '.*\.wav$';
inFiles = findFiles(estimateDir, inFilesOrPattern);
OPS_score = zeros(length(inFiles),1);
for i = 1:length(inFiles);
    inFile_org = strrep(inFiles{i},environment,'ORG');
    a = fullfile(originalDir,inFile_org);
    originalFiles = {a};
    estimateFile = fullfile(estimateDir, inFiles{i});
    options.destDir = tempDir;
    options.segmentationFactor = 1; % increase this integer if you experienced "out of memory" problems
    res = PEASS_ObjectiveMeasure(originalFiles,estimateFile,options);
    OPS_score(i) =res.OPS;
end
score = mean(OPS_score)
standard_deviation = std(OPS_score);
standard_err_of_mean = standard_deviation/sqrt(length(inFiles));
save(resultDir,'OPS_score','score','standard_err_of_mean');
end