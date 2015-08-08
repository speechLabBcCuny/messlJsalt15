function Generate_reverbTestMix2(overwrite, outDir)
% Re-write of official Reverb dev and eval data generation script to use two
% simultaneous talkers at different locations in the same room.

if ~exist('outDir', 'var') || isempty(outDir), outDir = '/export/ws15-ffs-data2/mmandel/data/reverb/mixes/'; end
if ~exist('overwrite', 'var') || isempty(overwrite), overwrite = false; end

baseDir = '/export/ws15-ffs-data2/mmandel/data/reverb';
wsjDir = fullfile(baseDir, 'wsjcam0');
rirAndNoiseDir = fullfile(baseDir, 'reverb_tools_for_Generate_SimData');
saveDir = fullfile(outDir, 'data/mc_train/');

if ~exist(fullfile(wsjDir,'data'), 'file')
   error(['Could not find wsjcam0 corpus : Please confirm if %s is ' ...
          'a correct path to your clean WSJCAM0 corpus'], wsjDir); 
end

% List of RIRs
rooms = {'SimRoom1', 'SimRoom2', 'SimRoom3'};
noises = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10'};
dists = {'near', 'far'};
angs = {'AnglA', 'AnglB'};

makeDataset('etc/audio_si_dt5a.lst', rooms, dists, 1, angs, 1, noises, ...
            wsjDir, rirAndNoiseDir, fullfile(outDir, 'dt/data/near_test'), ...
            overwrite)
makeDataset('etc/audio_si_dt5a.lst', rooms, dists, 2, angs, 1, noises, ...
            wsjDir, rirAndNoiseDir, fullfile(outDir, 'dt/data/far_test'), ...
            overwrite)
makeDataset('etc/audio_si_et_1.lst', rooms, dists, 1, angs, 2, noises, ...
            wsjDir, rirAndNoiseDir, fullfile(outDir, 'et/data/near_test'), ...
            overwrite)
makeDataset('etc/audio_si_et_1.lst', rooms, dists, 2, angs, 2, noises, ...
            wsjDir, rirAndNoiseDir, fullfile(outDir, 'et/data/far_test'), ...
            overwrite)


function makeDataset(flist, rooms, dists, useDists, angs, useAngs, noises, ...
                     wsjDir, rirAndNoiseDir, saveDir, overwrite)

flist = fullfile(rirAndNoiseDir, flist);

% Parameters related to acoustic conditions
SNRdB = 20;
gain = 0.25;
fs = 16000;
otherFStride = 2267;   % Prime to pseudo-randomize other utterance

Na = length(useAngs);
Nd = length(useDists);
Nr = length(rooms);
Nn = length(noises);
Nch = 8;

disp('Loading RIRs...')
for r = 1:length(rooms)
    for d = 1:length(dists)
        for a = 1:length(angs)
            rirFile{r,d,a} = fullfile(rirAndNoiseDir, 'RIR', ['RIR_' rooms{r} '_' ...
                                dists{d} '_' angs{a} '.wav']);
            rir{r,d,a} = wavread(rirFile{r,d,a});

            [~,delay{r,d,a}] = max(rir{r,d,a},[],1);
            before_impulse = floor(fs*0.001);
            after_impulse = floor(fs*0.05);
            for ch = 1:size(rir{r,d,a},2)
                inds = delay{r,d,a}(ch)-before_impulse : delay{r,d,a}(ch)+after_impulse;
                rirDirect{r,d,a}(:,ch) = rir{r,d,a}(inds,ch);
            end
        end
    end
end


files = readLines(flist);
lastNoiseFile = '';
otherF = round(length(files)/2);
for f = 1:length(files)
    fprintf('%d: %s\n', f, files{f});

    baseOutFile = fullfile(saveDir, files{f});
    doneFile = [baseOutFile '.done'];
    if exist(doneFile, 'file') && ~overwrite
        fprintf('\b <-- Skipping\n');
        continue
    end

    % Figure out what RIR and noise to use to be consistent
    % with original REVERB data
    a = mod(f-1, Na) + 1;
    d = floor(mod(f-1, Na*Nd)/Na) + 1;
    r = floor(mod(f-1, Na*Nd*Nr)/(Na*Nd)) + 1;
    n = floor(mod(f-1, Na*Nd*Nr*Nn)/(Na*Nd*Nr)) + 1;
    c = floor(mod(f-1, Na*Nd*Nr*Nch)/(Na*Nd*Nr)) + 1;

    a = useAngs(a);
    d = useDists(d);

    noiseFile = fullfile(rirAndNoiseDir, 'NOISE', ['Noise_' rooms{r}, ...
                        '_' noises{n} '.wav']);
    if ~strcmp(noiseFile, lastNoiseFile)
        % fprintf('Loading noise: %s\n', noiseFile);
        noise = wavread(noiseFile);
        lastNoiseFile = noiseFile;
    end

    sphFile = fullfile(wsjDir, 'data', [files{f} '.wv1']);
    [x Px] = loadAndSpatialize(sphFile, rir{r,d,a}, rirDirect{r,d,a}, c);
    scaledNoise = scaleNoise(noise, size(x,1), Px, SNRdB, c);

    saveWav(x, gain, delay{r,d,a}, c, fs, [baseOutFile '_src1.wav']);
    saveWav(x + scaledNoise, gain, delay{r,d,a}, c, fs, [baseOutFile '_src1n.wav']);

    % Put utterances from different talkers at other positions in the same room
    for od = 1:length(dists)
        for oa = 1:length(angs)
            if (od == d) && (oa == a)
                % Skip same exact location
                continue;
            end
            otherF = mod(otherF + otherFStride - 1, length(files)) + 1;
            while speakerName(files{f}) == speakerName(files{otherF})
                otherF = mod(otherF + otherFStride - 1, length(files)) + 1;
            end

            sphFile2 = fullfile(wsjDir, 'data', [files{otherF} '.wv1']);
            [x2 Px2] = loadAndSpatialize(sphFile2, rir{r,od,oa}, rirDirect{r,od,oa},c);
            x2 = zeroPadOrTrim(x2, size(x,1));

            mixNum = oa-1+2*(od-1);
            mixFile = sprintf('%s_mix%d', baseOutFile, mixNum);
            saveWav(x + x2 + scaledNoise, gain, delay{r,d,a}, c, fs, ...
                    [mixFile '.wav']);
            saveMat([mixFile '.mat'], sphFile, sphFile2, noiseFile, ...
                    r,d,a,c, delay, rirFile);
        end
    end

    % Touch doneFile
    fclose(fopen(doneFile, 'w'));
end


function [y Px] = loadAndSpatialize(sphFile, rir, rirDirect, ch)
x = sph2wav(sphFile);
y = fconv(x, rir);
direct_signal = fconv(x, rirDirect(:,ch));
Px = diag(mean(direct_signal.^2,1));


function scaledNoise = scaleNoise(noise, lenX, Px, SNRdB, ch)
noise = noise(1:lenX,:);
noiseRef = noise(:,ch);
iPn = diag(1./mean(noiseRef.^2,1));
Msnr = sqrt(10^(-SNRdB/10)*iPn*Px);
scaledNoise = noise*Msnr;


function spk = speakerName(sphFile)
% Speaker name is first three characters of file name without any directories
bn = basename(sphFile);
spk = bn(1:3);

function x = zeroPadOrTrim(x, len)
if size(x,1) > len
    x = x(1:len,:);
elseif size(x,1) < len
    x = [x; zeros(len-size(x,1), size(x,2))];
end

function fnames = readLines(flist)
fid = fopen(flist);
fnames = {};
while 1
    fnameTmp = fgetl(fid);
    if ~ischar(fnameTmp), break; end
    fnames{end+1} = fnameTmp;
end
fclose(fid);

function saveWav(x, gain, delay, c, fs, outFile)
y = gain * x(delay(c):end,:);
y = y(:,[c:end 1:c-1]);
fprintf('Writing: %s\n', outFile);
ensureDirExists(outFile);
wavwrite(y, fs, outFile);

function saveMat(matFile, sphFile, sphFile2, noiseFile, r,d,a,c, delay, rirFile)
delay = delay{r,d,a}(c);
rirFile = rirFile{r,d,a};
save(matFile, 'sphFile', 'sphFile2', 'noiseFile', 'r', 'd', 'a', 'c', ...
     'delay', 'rirFile');


%%%%
function [y]=fconv(x, h)
%FCONV Fast Convolution
%   [y] = FCONV(x, h) convolves x and h
%
%      x = input vector
%      h = input vector
% 
%      See also CONV
%
%   NOTES:
%
%   1) I have a short article explaining what a convolution is.  It
%      is available at http://stevem.us/fconv.html.
%
%
%Version 1.0
%Coded by: Stephen G. McGovern, 2003-2004.
%
%Copyright (c) 2003, Stephen McGovern
%All rights reserved.
%
%THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
%AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
%IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
%ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
%LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
%CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
%SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
%INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
%CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
%ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
%POSSIBILITY OF SUCH DAMAGE.

Ly=length(x)+length(h)-1;
X=fft(x, Ly);
H=fft(h, Ly);
Y=bsxfun(@times, X, H);
y=real(ifft(Y));
