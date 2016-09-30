function [Y data] = stubI_Masks(X, C, mode, fail, fs, inFile, I, refMic, d, useHardMask, beamformer, varargin)
    C2 = repmat(C,1,1,1);
    switch mode
        case 'ideal_amplitude'
            data.mask = abs(X)/abs(C2);
        case 'phase_sensitive'
            data.mask = real(X/C2);
        case 'ideal_copmlex'
            data.mask = X/C;
    end