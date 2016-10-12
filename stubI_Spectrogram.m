function [Y, data] = stubI_Spectrogram(X)
    data.input=X;
    % clean speech is just C;
    Y=X(:,:,1);
end