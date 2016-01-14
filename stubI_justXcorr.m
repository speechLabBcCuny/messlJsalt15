function [Y data] = stubI_justXcorr(X, fail, fs, inFile, maxDelay)

% Just compute broadband cross-correlations between channels, don't
% actually modify the signals at all.

if ~exist('maxDelay', 'var') || isempty(maxDelay), maxDelay = 1000; end

nsamp = numel(X(:,:,1)) - 1024;
x = istft_multi(X, nsamp).';

tau = -maxDelay:maxDelay;
for i = 1:size(X,3)
    for j = i+1:size(X,3)
        % [tdoa(i,j,:) masks failure(i,j)] = phatLoc(X(:,:,[i j]), tau, 3, 1024, 1);
        [r lags] = xcorr(x(:,i), x(:,j), maxDelay);
        if i == 1, plot(lags, r); drawnow, end
        tdoa(i,j) = lags(argmax(abs(r)));
    end
end

Y = X;
%data = struct('tau', tau, 'tdoa', tdoa, 'failure', failure);
data = struct('lags', lags, 'tdoa', tdoa);
1+1;