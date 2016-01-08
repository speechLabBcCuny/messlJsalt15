function correctCh0Delay(wavDir, dataDir)

files = findFiles(wavDir, '.*.CH0.wav');

for f = 1:length(files)
    wavFile = fullfile(wavDir, files{f});
    bakFile = fullfile(wavDir, strrep(files{f}, 'CH0.', 'CH0_nodelay.'));
    dataFile = fullfile(dataDir, strrep(files{f}, 'CH0.wav', 'mat'));

    if ~exist(bakFile, 'file')
        copyfile(wavFile, bakFile)
    end
    
    d = load(dataFile);
    tdoas = -d.data.tdoa(1,2:end)
    tdoa = round(median(tdoas))
    if sum(abs(tdoas - tdoa) > 30) > 1
        warning('correctCh0Delay:medianTdoa', 'Median TDOA very different from multiple mics')
    end
    
    [x fs] = audioread(bakFile);
    x = [zeros(tdoa,size(x,2)); x(1:end-tdoa,:)];
    
    delete(wavFile);
    audiowrite(wavFile, x, fs);
end
