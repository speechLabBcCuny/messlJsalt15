function Xp = pickChanWithBestSnr(X, mask, fail)

% Figure out what to apply the mask to
% Stupidest way: pick the channel with best estimated SNR

P = magSq(X);
Xp = zeros(size(X,1), size(X,2), size(mask,3));
for s = 1:size(mask,3)
    signal = squeeze(sum(sum(bsxfun(@times, P,   mask(:,:,s)), 1), 2));
    noise  = squeeze(sum(sum(bsxfun(@times, P, 1-mask(:,:,s)), 1), 2));
    snr = signal ./ noise - 1e9*fail';
    
    [~,bestChan] = max(snr);
    Xp(:,:,s) = X(:,:,bestChan);
end
