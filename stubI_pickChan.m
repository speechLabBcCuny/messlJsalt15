function [Y mask] = stubI_pickChan(X, fail, fs, inFiles, chan)

if ~exist('chan', 'var') || isempty(chan), chan = 1; end

Y = X(:,:,chan);
mask = ones(size(Y));
