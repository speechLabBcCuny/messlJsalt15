function correctCh0Delay(wavDir, dataDir)

files = findFiles(wavDir, '.*.CH0.wav');

for f = 1:length(files)
    fprintf('%d: %s\n', f, files{f});
    
    wavFile = fullfile(wavDir, files{f});
    bakFile = fullfile(wavDir, strrep(files{f}, 'CH0.', 'CH0_origdelay.'));
    dataFile = fullfile(dataDir, strrep(files{f}, 'CH0.wav', 'mat'));

    if ~exist(dataFile, 'file')
        fprintf('  ^^^ Skipping: no data file ^^^\n');
        continue
    end
    if ~exist(bakFile, 'file')
        copyfile(wavFile, bakFile)
    end
    
    d = load(dataFile);
    tdoas = d.data.tdoa(1,2:end);
    tdoa = round(median(tdoas));
    fprintf('\b  TDOA: %d -- ', tdoa);
    fprintf('%d ', tdoas);
    fprintf('\n')
    if sum(abs(tdoas - tdoa) > 30) > 1
        warning('correctCh0Delay:medianTdoa', 'Median TDOA very different from multiple mics')
    end
    
    [x fs] = audioread(bakFile);

    if tdoa > 0
        x = [x(1+tdoa:end,:); zeros(tdoa,size(x,2))];
    else
        x = [zeros(-tdoa,size(x,2)); x(1:end+tdoa,:)];
    end
    
    delete(wavFile);
    audiowrite(wavFile, x, fs);
end
