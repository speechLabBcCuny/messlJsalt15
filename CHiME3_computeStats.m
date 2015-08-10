function CHiME3_computeStats(outDir, part, overwrite)

% Save various auxiliary information computed by chime wrapper
%
% Based on the baseline enhancement script.  Data will be written in a
% standard directory structure rooted at outDir.
% 
% Part is a tuple [n N] meaning process the nth of N sets of utterances to
% allow for easy parallelization across matlab sessions.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright 2015 Michael Mandel
%                University of Sheffield (Jon Barker, Ricard Marxer)
%                Inria (Emmanuel Vincent)
%                Mitsubishi Electric Research Labs (Shinji Watanabe)
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%dataDir = '/export/ws15-ffs-data/corpora/chime3/CHiME3/data';
dataDir = '/export/ws15-ffs-data2/mmandel/data/chime3/CHiME3';

if ~exist('outDir', 'var') || isempty(outDir), 
    outDir=fullfile(dataDir, 'stats/'); % path to enhanced utterances
end
if ~exist('overwrite', 'var') || isempty(overwrite), overwrite = false; end
if ~exist('part', 'var') || isempty(part), part = [1 1]; end
if ~exist('ignoreErrors', 'var') || isempty(ignoreErrors), ignoreErrors = false; end

%addpath ../utils;
upath=fullfile(dataDir, 'audio/16kHz/isolated/'); % path to segmented utterances
cpath=fullfile(dataDir, 'audio/16kHz/embedded/'); % path to continuous recordings
bpath=fullfile(dataDir, 'audio/16kHz/backgrounds/'); % path to noise backgrounds
apath=fullfile(dataDir, 'annotations/'); % path to JSON annotations
nchan=6;

% Define hyper-parameters
pow_thresh=-20; % threshold in dB below which a microphone is considered to fail
wlen = 1024; % STFT window length
cmin=6400; % minimum context duration (400 ms)
cmax=12800; % maximum context duration (800 ms)

%sets={'tr05' 'dt05' 'et05'};
%modes={'real' 'simu'};
sets={'dt05' 'et05'};
modes={'real'};
for set_ind=1:length(sets),
    set=sets{set_ind};
    for mode_ind=1:length(modes),
        mode=modes{mode_ind};
        
        % Read annotations
        mat=json2mat([apath set '_' mode '.json']);
        real_mat=json2mat([apath set '_real.json']);
            
        for utt_ind=part(1):part(2):length(mat),
            udir=[upath set '_' lower(mat{utt_ind}.environment) '_' mode '/'];
            edir=[outDir set '_' lower(mat{utt_ind}.environment) '_' mode '/'];
            uname=[mat{utt_ind}.speaker '_' mat{utt_ind}.wsj_name '_' mat{utt_ind}.environment];
            outFile = [edir uname '.mat'];
            
            if exist(outFile, 'file') && ~overwrite
                fprintf('Skipping %s\n', outFile);
                continue
            else
                fprintf('%d: %s\n', utt_ind, outFile);
            end
            
            % Load WAV files
            xsize=wavread([udir uname '.CH1.wav'],'size');
            nsampl=xsize(1);
            x=zeros(nsampl,nchan);
            for c=1:nchan,
                [x(:,c),fs]=wavread([udir uname '.CH' int2str(c) '.wav']);
            end
            
            % Check microphone failure
            xpow=sum(x.^2,1);
            xpow=10*log10(xpow/max(xpow));
            fail=(xpow<=pow_thresh);
            
            % Load context (up to 5 s immediately preceding the utterance)
            if strcmp(mode,'real'),
                cname=mat{utt_ind}.wavfile;
                cbeg=round(mat{utt_ind}.start*fs)-cmax;
                cend=round(mat{utt_ind}.start*fs)-1;
                for utt_ind_over=1:length(mat),
                    cend_over=round(mat{utt_ind_over}.end*fs);
                    if strcmp(mat{utt_ind_over}.wavfile,cname) && (cend_over >= cbeg) && (cend_over < cend),
                        cbeg=cend_over+1;
                    end
                end
                cbeg=min(cbeg,cend-cmin);
                n=zeros(cend-cbeg+1,nchan);
                for c=1:nchan,
                    n(:,c)=wavread([cpath cname '.CH' int2str(c) '.wav'],[cbeg cend]);
                end
            elseif strcmp(set,'tr05'),
                cname=mat{utt_ind}.noise_wavfile;
                cbeg=round(mat{utt_ind}.noise_start*fs)-cmax;
                cend=round(mat{utt_ind}.noise_start*fs)-1;
                if cbeg < 1  % Added by MIM
                    warning('cbeg < 1: %d, setting to 1', cbeg)
                    cbeg = 1;
                end
                n=zeros(cend-cbeg+1,nchan);
                for c=1:nchan,
                    n(:,c)=wavread([bpath cname '.CH' int2str(c) '.wav'],[cbeg cend]);
                end
            else
                cname=mat{utt_ind}.noise_wavfile;
                cbeg=round(mat{utt_ind}.noise_start*fs)-cmax;
                cend=round(mat{utt_ind}.noise_start*fs)-1;
                for utt_ind_over=1:length(real_mat),
                    cend_over=round(real_mat{utt_ind_over}.end*fs);
                    if strcmp(mat{utt_ind_over}.wavfile,cname) && (cend_over >= cbeg) && (cend_over < cend),
                        cbeg=cend_over+1;
                    end
                end
                cbeg=min(cbeg,cend-cmin);
                n=zeros(cend-cbeg+1,nchan);
                for c=1:nchan,
                    n(:,c)=wavread([cpath cname '.CH' int2str(c) '.wav'],[cbeg cend]);
                end
            end
    
            % STFT
            X = stft_multi(x.',wlen);
            [nbin,nfram,~] = size(X);
            
            % Compute noise covariance matrix
            N=stft_multi(n.',wlen);
            Ncov=zeros(nchan,nchan,nbin);
            for f=1:nbin,
                for n=1:size(N,2),
                    Ntf=permute(N(f,n,:),[3 1 2]);
                    Ncov(:,:,f)=Ncov(:,:,f)+Ntf*Ntf';
                end
                Ncov(:,:,f)=Ncov(:,:,f)/size(N,2);
            end
            
            % Localize and track the speaker
            [~,TDOA]=localize(X);

            if ~exist(edir,'dir'),
                system(['mkdir -p ' edir]);
            end
            save(outFile, 'Ncov', 'fail', 'TDOA', 'fs', 'nchan', ...
                 'nbin', 'nsampl', 'nfram');
        end
    end
end

return