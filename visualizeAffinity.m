function visualizeAffinity(X, affinity)

% Interactively visualize a cosine affinity matrix created by an embedding

[F T S] = size(affinity);
A = reshape(permute(affinity, [3 1 2]), S, F*T);
A = bsxfun(@rdivide, A, sqrt(sum(A.^2,1)));

subplots({X, X})
while true
    % Get click from user
    [xc,yc,bc] = ginput(1);
    if bc ~= 1
        break
    end
    if isempty(xc) || (xc < 1) || (yc < 1) || (xc > T) || (yc > F)
        continue;
    end
    
    ft = (round(xc)-1)*F+1 + round(yc)-1;
    Xa = reshape(A(:,ft)' * A, F, T);
    
    subplots({X, Xa})
    hold on, plot(xc, yc, 'o'), hold off
end
