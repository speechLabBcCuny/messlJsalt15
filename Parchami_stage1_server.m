
inDir = '/home/vietanh/data/CHiME3/data/audio/16kHz/isolated/dt05_str_simu';
outDir = '/home/vietanh/speech_class_16/Parchami_stage1/dt05_str_simu';
inFilesOrPattern = '.*\.CH1\.wav$';
% search file in input directory, inFiles is a matrix contains file names such as bus.wav
inFiles = findFiles(inDir, inFilesOrPattern); % inFiles look like is m x 1 cell, example 1 cell: F02_01BO030M_CAF.CH0.wav
for h = 1:length(inFiles);
    
    inFile = fullfile(inDir, inFiles{h});
    inFileNoCh = strrep(inFiles{h},'.CH1','');
    outWavFile = fullfile(outDir,'wavout', inFileNoCh);
    [inD inF inE] = fileparts(inFile);
    info = audioinfo(inFile);
    fs = info.SampleRate;
    x = zeros(info.TotalSamples,1,6);
    for i = 1 : 6;
        chanFile = fullfile(inD, [strrep(inF, '.CH1', sprintf('.CH%d', i)) inE]);
        noisy_sampling = audioread(chanFile); 
        x(:,:,i) = noisy_sampling;
    end
    
    outMaskFile = fullfile(outDir, 'data', strrep(inFileNoCh, '.wav', '.mat'));
    nsample = size(noisy_sampling,1);
    x = permute(x,[3 1 2]);
    
    X = stft_multi(x,1024);
    %save(outMaskFile,'X');
    [F T C] = size(X);
    X = permute(X,[3,1,2]);  % now X is C F T
    alpha_min = 0.25;
    alpha = alpha_min * ones(F,T+4);
    alpha_max = 0.94;
    minCor = 1;
    beta = 0.65;
    beta_MVDR = 0;
    w = [0.5437 0.2956 0.1607]; % D = 3, w[i] = 0.5437 ^ i
    TH = 22;
    TL = 0.35;
    pickMic = [0;0;0;0;1;0];
    % Choose weigh = 0.92 to calculate Noisy covariance matrix
    % as in equation (3) of Souden, M., Chen, J., Benesty, J., & Affes, S. (2011).
    % An integrated solution for online multichannel noise tracking and reduction.
    % IEEE Transactions on Audio, Speech, and Language Processing, 19(7), 2159-2169.
    
    weight = 0.92;
    Mcov = zeros(C,C,F,T);
    
    for f = 1:F
        Mcov(:,:,f,1) = (1-weight) * X(:,f,1)* X(:,f,1)';
    end
    for t = 2:T
        for f=1:F
            Mcov(:,:,f,t) = weight* Mcov(:,:,f,t-1) + (1-weight)*X(:,f,t)*X(:,f,t)';
        end
    end
    
    Mcov1 = padarray(Mcov,[0 0 0 1],'pre');
    Mcov_pad = padarray(Mcov1,[0 0 0 3],'post');
    
    P = zeros(C,C,F,T+4);
    % Calculate P(k,l) in Parchami paper, using equation (4), using 2 iteration
    % Estimate for special case P(k,1)
    X1 = padarray(X,[0 0 1],'pre'); % pad 0 to front of X for X(:,0)
    X_pad = padarray(X1,[0 0 3],'post');% pad 0 (3 times) to post of X for X(:,T+1);...X(:,T+3)
    
    for t = 2:T+1
        for f=1:F
            for j = 1:2
                sum = w(1)*X_pad(:,f,t+1)*X_pad(:,f,t+1)' + w(2)* X_pad(:,f,t+2)*X_pad(:,f,t+2)'+ w(3)*X_pad(:,f,t+3)*X_pad(:,f,t+3)';
                P(:,:,f,t) = beta*alpha(f,t)* P(:,:,f,t-1)+(1-alpha(f,t))*X_pad(:,f,t)* X_pad(:,f,t)'+(1-beta)*alpha(f,t)*sum ;
                zeta(f,t) = norm(Mcov_pad(:,:,f,t)- P(:,:,f,t)) * inv(norm(P(:,:,f,t)));
                if zeta(f,t) >= TH
                    zeta_norm(f,t) = 1;
                elseif TL < zeta(f,t) < TH
                    zeta_norm(f,t) = (zeta(f,t)-TL)/(TH-TL);
                else
                    zeta_norm(f,t) = 0;
                end
                alpha(f,t) = alpha_min + (alpha_max - alpha_min)*zeta_norm(f,t);
            end
        end
    end
    
    Ncov = P(:,:,:,2:end-3);
    % M = 2;
    % for i = 1:C
    %     for j = 1:C
    %         for f = 1:F
    %             for t = M:T
    %                 Q = Ncov(i,j,f,t-M+1:t);
    %                 mi = min(Q);
    %                 Ncov(i,j,f,t) = mi;
    %             end
    %         end
    %     end
    % end
    % Ncov = (1 + (M-1)/2) * Ncov;
    Y  = zeros(F,T);
    for t = 1:T
        for f = 1:F
            RNcov = Ncov(:,:,f,t) ;
            RMcov = Mcov(:,:,f,t);
            num = trace(RNcov\RMcov) - eye(C);
            lambda = real(trace(num));
            den = beta_MVDR + lambda;
            h = (num * pickMic) / den;
            Y(f,t) = h' * X(:,f,t);
        end
    end
    y = istft_multi(Y,nsample);
    audiowrite(outWavFile,y',fs);
end
    