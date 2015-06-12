function [sepPaths sdr sir sar] = evalMerl(sepDir, refDir, overwrite)

% Evaluate Scott's files.  Uses file names to look up appropriate
% references.  Assumes that reference file names are globally unique.
% Evaluates using BSS_EVAL.

if ~exist('overwrite', 'var') || isempty(overwrite), overwrite = false; end

outFile = fullfile(sepDir, 'bss_eval_res.mat');
if exist(outFile, 'file') && ~overwrite
    fprintf('Loading from %s...\n', outFile);
    load(outFile)
    return
end

[sepFiles,sepPaths] = findFiles(sepDir, '.*.wav');
[refFiles,refPaths] = findFiles(refDir, '.*.wav');
refBasenames = listMap(@(x) basename(x,0), refFiles);

for f = 1:length(sepFiles)
    fprintf('%d: %s\n', f, sepFiles{f});
    [se fs] = wavread(sepPaths{f});
    se = se(:,end-1:-1:1);  % Chop off garbage source, re-order
    
    sepName = basename(sepFiles{f}, 0);
    cleanNames = split(sepName, '_');
    s = zeros(size(se,1), length(cleanNames));
    for c = 1:length(cleanNames)
        ind = find(strcmp(cleanNames{c}, refBasenames));
        assert(length(ind) == 1);
        assert(strcmp(basename(refPaths{ind}, 0), cleanNames{c}));

        [x fsr] = wavread(refPaths{ind});
        assert(size(x,2) == 1)
        assert(fsr == fs)
        
        s(1:length(x),c) = x;
    end
    
    [sdr(f,:) sir(f,:) sar(f,:)] = bss_eval_sources_nosort(se', s');
    % [sdr(f,:) sir(f,:) sar(f,:)] = bss_eval_sources(se', s');
end

save(outFile, 'sepPaths', 'refPaths', 'sdr', 'sir', 'sar')
