addpath('../../mimlib');
addpath('../../utils');
sdrr =0.0;
opss = 0.0;
sirr = 0.0;
sarr = 0.0;
isrr = 0.0;
estimate_dir = '/scratch/near/replayMessl/wav/';
ref_dir = '/home/data/CHiME3/data/audio/16kHz/local/messl-mvdr-output/wav/';
estimate_list = findFiles(estimate_dir,'dt05.*simu.*\.wav');
ref_list = findFiles(ref_dir,'dt05.*simu.*\.wav');
for f = 1:length(estimate_list)
    estimate_file = fullfile(estimate_dir,estimate_list{f});
    ref_file = fullfile(ref_dir,ref_list{f});
    options.destDir = '/scratch/near/peass_temp/';
    options.segmentationFactor = 1; % increase this integer if you experienced "out of memory" problems
    res = PEASS_ObjectiveMeasure({ref_file},estimate_file,options);
fprintf(' - SDR = %.1f dB\n - ISR = %.1f dB\n - SIR = %.1f dB\n - SAR = %.1f dB\n - OPS = %.1f dB\n',res.SDR,res.ISR,res.SIR,res.SAR,res.OPS);
    sdrr = sdrr+ res.SDR;
    opss = opss+res.OPS;
    sirr = sirr + res.SIR;
    isrr = isrr +res.ISR;
    sarr = sarr + res.SAR;

end;
sarr = sarr/length(estimate_list);
isrr = isrr/length(estimate_list);
sirr = sirr/length(estimate_list);
opss = opss/length(estimate_list);
sdrr = sdrr/length(estimate_list);
fprintf('the average sdr is %f',sdrr);
fprintf('the average ops is %f',opss);
fprintf('the average sir is %f',sirr);
fprintf('the average isr is %f',isrr);
fprintf('the average sar is %f',sarr);
