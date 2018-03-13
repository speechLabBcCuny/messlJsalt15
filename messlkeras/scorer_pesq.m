function a = scorer_pesq(enhanced_dir)

% compute  per file val-loss PESQ,( WER, PESQ, SDR)

% for CHIME3 Real/Simu

addpath('/home/felix/Research/mimlib');
addpath('/home/felix/Research/utils');
addpath('/home/felix/Research/messlJsalt15/pesq')




% references files
ref_dir = '/scratch/felix/MESSLKERAS/CHiME3-MVDR/wav/'

ref_list_all = findFiles(ref_dir,'.*\.wav');

ref_list_real = findFiles(ref_dir,'[de]t05.*real.*\.wav');

ref_list_simu = findFiles(ref_dir,'[de]t05.*simu.*\.wav');

% to be analyzed
% enhanced_dir = '/scratch/felix/MESSLKERAS/CHiME3-exps/2018-03-09_19:16:57/iaf_2018-03-09_19:29:56/enhanced-audio-final-model/average/wav/'

enhanced_list_real = findFiles(enhanced_dir,'[de]t05.*real.*\.wav');

enhanced_list_simu = findFiles(enhanced_dir,'[de]t05.*simu.*\.wav');

% directory in which to save results
% base folder
base_results_dir = strsplit(enhanced_dir, '/');
base_results_dir = strjoin(base_results_dir(1:end-2), '/');

disp(base_results_dir)

pesq_results_dir = fullfile(base_results_dir, '/scores-pesq/');

exist(pesq_results_dir)

% run PESQ for all real files, and save results
real_result_list = [];

for f = 1:length(enhanced_list_real)
    % files to compare
    enhanced_file_path = fullfile(enhanced_dir, enhanced_list_real{f});
    reference_file_path = fullfile(ref_dir, ref_list_real{f} );
    
    % evaluate PESQ
    pesq_score = pesq(reference_file_path, enhanced_file_path, 160);
    % add to list
    real_result_list(f) = pesq_score;

    % print result 
    fprintf('file: %s\n', string(enhanced_list_real{f}));
    fprintf('pesq score: %f\n', pesq_score);
    
    % save results in appropriate folder
    % make sub-dirs
    save_file_path = fullfile(pesq_results_dir, enhanced_list_real{f});
    ensureDirExists(save_file_path);
    % save results to txt file
    save_file = fopen(strcat(save_file_path,'.pesq'),'w');
    fprintf(save_file, '%f\n', pesq_score);
    
end;


% run PESQ for all simu files, and save results
real_result_list = [];

for f = 1:length(enhanced_list_simu)
    % files to compare
    enhanced_file_path = fullfile(enhanced_dir, enhanced_list_simu{f});
    reference_file_path = fullfile(ref_dir, ref_list_simu{f} );
    
    % evaluate PESQ
    pesq_score = pesq(reference_file_path, enhanced_file_path, 160);
    % add to list
    real_result_list(f) = pesq_score;

    % print result 
    fprintf('file: %s\n', string(enhanced_list_simu{f}));
    fprintf('pesq score: %f\n', pesq_score);
    
    % save results in appropriate folder
    % make sub-dirs
    save_file_path = fullfile(pesq_results_dir, enhanced_list_simu{f});
    ensureDirExists(save_file_path);
    % save results to txt file
    save_file = fopen(strcat(save_file_path,'.pesq'),'w');
    fprintf(save_file, '%f\n', pesq_score);
    
end;


% CH5 file list
noisy_dir = '/home/data/CHiME3/data/audio/16kHz/isolated/';
ch5_list = findFiles(noisy_dir,'.*CH5\.wav');

% check that the the files order matches
disp(ch5_list(1:10))
disp(ref_list_all(1:10))
disp(ch5_list(end-10:end))
disp(ref_list_all(end-10:end))

% BASESLINE PESQ for CH5 

% run PESQ for all CH5 files, and save results
ch5_result_list = [];
pesq_results_dir = '/home/proj/MESSLKERAS/Baselines/CH5-PESQ/';

for f = 1:length(ch5_list)
    % files to compare
    noisy_file_path = fullfile(noisy_dir, ch5_list{f});
    reference_file_path = fullfile(ref_dir, ref_list_all{f} );
    
    % evaluate PESQ
    pesq_score = pesq(reference_file_path, noisy_file_path, 160);
    % add to list
    ch5_result_list(f) = pesq_score;

    % print result 
    %fprintf('file: %s\n', string(ch5_list{f}));
    %fprintf('pesq score: %f\n', pesq_score);
    
    % save results in appropriate folder
    % make sub-dirs
    save_file_path = fullfile(pesq_results_dir, ch5_list{f});
    %disp(save_file_path)
    
    ensureDirExists(save_file_path);
    % save results to txt file
    save_file = fopen(strcat(save_file_path,'.pesq'),'w');
    fprintf(save_file, '%f\n', pesq_score);
    
end;