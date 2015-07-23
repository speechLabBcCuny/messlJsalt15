function trimMerlRirs()

% Get rid of noise-floor tail in Scott's RIRs and noisy lead-in

inFiles = {
    '/home/data/merlTest/rir/orig/h_angA_far.wav'
    '/home/data/merlTest/rir/orig/h_angA_near.wav'
    '/home/data/merlTest/rir/orig/h_angB_far.wav'
    '/home/data/merlTest/rir/orig/h_angB_near.wav'
};

onset_s = [0.137 0.133 0.137 0.133];
lateReverbOnset_s = [0.165 0.15 0.165 0.15];
lateReverbEndLinear_s = [0.22 0.22 0.21 0.22];
offset_s = [0.28 0.27 0.27 0.27];
keepAtBegin_s = 0.01;
startTime_s = min(onset_s) - keepAtBegin_s;
onRiseTime_s = 0.005;

for f = 1:length(inFiles)
    [x fs] = wavread(inFiles{f});
    t = (1:size(x,1)) / fs;    

    rt60 = estimateRt60(x, fs, lateReverbOnset_s(f), lateReverbEndLinear_s(f));
    win1 = makeFadeIn(size(x), fs, onset_s(f)-onRiseTime_s, onset_s(f));
    win2 = makeDecay(size(x), fs, offset_s(f), 4*mean(rt60));    
    xw = x .* win1 .* win2;
    plot(t, db(abs(x)), t, db(abs(xw)));
    
    xw = xw(round(startTime_s*fs):end,:);
    wavWriteBetter(xw, fs, strrep(inFiles{f}, '/orig/', '/trimmed/'));

    win1 = makeFadeIn(size(x), fs, lateReverbOnset_s(f)-onRiseTime_s, lateReverbOnset_s(f));
    xw = x .* win1 .* win2;
    xw = xw(round(startTime_s*fs):end,:);
    wavWriteBetter(xw, fs, strrep(inFiles{f}, '/orig/', '/diffuse/'));
end

makeDecay([16000 2], 16000, 0.5, 0.1);


function rt60 = estimateRt60(x, fs, start_s, stop_s)
% Schroeder integral to estimate RT60 time
start = round(start_s * fs);
stop = round(stop_s * fs);
sch = db(flipud(cumsum(flipud(x.^2))));
rt60 = (stop_s - start_s) ./ (sch(start,:) - sch(stop,:)) * 60;
fprintf('Mean RT60: %g\n', mean(rt60));

function win = makeDecay(sz, fs, start_s, rt60)
win = ones(sz);
t = (1:sz(1))' / fs;
decay = 10.^(-60/20 * (t - start_s) / rt60);
win = bsxfun(@min, win, decay);
%plot(t, db(win))

function win = makeFadeIn(sz, fs, start_s, stop_s)
t = (1:sz(1))' / fs;
win = lim((t - start_s) / (stop_s - start_s), 0, 1);
win = repmat(win, 1, sz(2));
