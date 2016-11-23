function [Y, data] = stubI_Clean(inFile, workDir)
    %load ground truth wav file to variable "clean"
    clean = audioread(strcat(workDir,'/wav/',strrep(inFile, '.CH1', '')));
    %compute the spectrogram of clean file
    C = stft_multi(clean.',1024);
    C = C(:,:,1);
    % repeat the spectrogram 7 times for build 7 masks
    C2 = repmat(C,1,1,6);
    
    data.clean = C2;
    % clean speech is just C;
    Y=C;
end