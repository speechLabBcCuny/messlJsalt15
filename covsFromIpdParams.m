function covs = covsFromIpdParams(X, ipdParams, N, tau, fs)

% Derive spatial covariance matrices from multichannel MESSL IPD parameters

nIpd = 25;

[F I nTau] = size(ipdParams(1).xi_wit);
assert(nTau == length(tau));
ipdParams = structCast(ipdParams, @isnumeric, @double);

channelPairs = nchoosek(1:N, 2);  % Brittle

covs = zeros(N, N, F+2, I);  % No IPD params for DC or Nyquist
for p = 1:size(channelPairs,1)
    cp = channelPairs(p,:);
    
    % Get probability of each IPD for each source at each frequency
    [prob phase] = visParams(ipdParams(p), tau, fs, 0, [], nIpd);

    % Trim off duplicate end point
    phase = phase(1:end-1);
    
    for s = 1:size(prob, 3)
        % Compute expected value of phase for each source at each freq
        expCpx = phase * prob(:,:,s) ./ max(sum(prob(:,:,s),1), 1e-9);
        
        % Project to unit norm (is this the right thing to do??)
        expPhase = expCpx ./ max(abs(expCpx), 1e-9);

        % Insert into appropriate entries in covariance
        covs(cp(1), cp(2), 2:end-1, s) = expPhase;
        covs(cp(2), cp(1), 2:end-1, s) = conj(expPhase);
    end
end

% Fill in diagonal
for c = 1:size(X,3)
    covs(c, c, :, :) = 1;
end

% for c = 1:size(X,3)
%     x2 = sqrt(mean(magSq(X(:,:,c)), 2));
%     covs(c,:,:,:) = bsxfun(@times, covs(c,:,:,:), permute(x2, [2 3 1]));
%     covs(:,c,:,:) = bsxfun(@times, covs(:,c,:,:), permute(x2, [2 3 1]));
% end
1+1;
