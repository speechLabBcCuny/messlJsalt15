function [Y, data] = stubI_Masks(X, fail, fs, inFile, workDir, mode, I, refMic, d, useHardMask, beamformer, varargin)
    %load ground truth wav file to clean
    clean = audioread(strcat(workDir,'/wav/',strrep(inFile, '.CH1', '')));
    %compute the spectrogram of clean file
    C = stft_multi(clean.',1024);
    C = C(:,:,1);
    % repeat the spectrogram 7 times for build 7 masks
    C2 = repmat(C,1,1,7);
    switch mode
        case 'ideal_amplitude'
            data.mask = abs(X)./abs(C2);
        case 'phase_sensitive'
            data.mask = real(X./C2);
        case 'ideal_complex'
            data.mask = X./C2;
    end
    % clean speech is just C;
    Y=C;
end