function [Y, data] = stubI_Spectrogram(X)
    data.input=X(:,:,2:7);
    % clean speech is just C;
    Y=X(:,:,1);
end