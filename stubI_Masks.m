function [Y, data] = stubI_Masks(X, inFile, workDir, mode)
    %load ground truth wav file to clean
    clean = audioread(strcat(workDir,'/wav/',strrep(inFile, '.CH1', '')));
    %compute the spectrogram of clean file
    C = stft_multi(clean.',1024);
    C = C(:,:,1);
    % repeat the spectrogram 7 times for build 7 masks
    C2 = repmat(C,1,1,7);
    
    % compute masks based on modes
    % formulas proposed in paper: 
%     Erdogan, Hakan, et al. 
%     "Phase-sensitive and recognition-boosted speech separation using deep recurrent neural networks." 
%     2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP).
    switch mode
        case 'ideal_amplitude'
            data.mask = lim(abs(db2mag(5)*C2)./abs(X), 0,1);
        case 'phase_sensitive'
            data.mask = lim(db2mag(5)*real(C2./X), -1, 1);
        case 'ideal_complex'
            data.mask = db2mag(5)*C2./X;
    end
    % clean speech is just C;
    Y=C;
end