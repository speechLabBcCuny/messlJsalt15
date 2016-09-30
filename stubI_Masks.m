function [Y data mask Xp] = stubI_Masks(X, fail, fs, inFile, mode, I, refMic, d, useHardMask, beamformer, varargin)
    clean = audioread('/Users/Near/Desktop/MESSL/mvdr_test/dev2/output/wav/F01_22GC010A_BTH.wav');
    C = stft_multi(clean.',1024);
    C = C(:,:,1);
    C2 = repmat(C,1,1,7);
    switch mode
        case 'ideal_amplitude'
            data.mask = abs(X)./abs(C2);
        case 'phase_sensitive'
            data.mask = real(X./C2);
        case 'ideal_complex'
            data.mask = X./C2;
    end
    Xp = zeros(size(X,1), size(X,2), size(data.mask,3));
for s = 1:size(data.mask,3)
        % No MVDR, pick a random channel
        Xp(:,:,s) = X(:,:,1);
end
Y = Xp .* data.mask;