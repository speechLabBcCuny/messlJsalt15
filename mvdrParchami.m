function Y= mvdrParchami(X,M)

% Perform MVDR beamforming with noise covariance matrix based on [1]. And
% MVDR formula based on[2] (equation 18 with beta = 0)
%
%  Y= mvdrParchami(X,M)
%
% [1]Parchami, M., Zhu, W. P., & Champagne, B. (2015, May).
% A new algorithm for noise PSD matrix estimation in multi-microphone
% speech enhancement based on recursive smoothing.
% [2] Souden, M., Benesty, J., & Affes, S. (2010).
% On optimal frequency-domain multichannel linear filtering for noise reduction.
%
% Inputs:
%   X:FxTxC tensor of complex input spectrograms for each channel,
%   M: number of past frames using for smoothing scheme (equation 9 of the
%   paper)
% Outputs:
%   Y: FxT matrix of estimated complex spectrum of target

[F T C] = size(X);
X = permute(X,[3,1,2]);  % now X is C F T
alpha_min = 0.25;
alpha = alpha_min * ones(F,T+4);
alpha_max = 0.94;
minCor = 1;
beta = 0.65;
beta_MVDR = 0;
w = [0.5437 0.2956 0.1607]; % D = 3, w[i] = 0.5437 ^ i
TH = 22;
TL = 0.35;
pickMic = [0;0;0;0;1;0];
% Choose weigh = 0.92 to calculate Noisy covariance matrix
% as in equation (3) of Souden, M., Chen, J., Benesty, J., & Affes, S. (2011).
% An integrated solution for online multichannel noise tracking and reduction.
% IEEE Transactions on Audio, Speech, and Language Processing, 19(7), 2159-2169.

weight = 0.92;
Mcov = zeros(C,C,F,T);

for f = 1:F
    Mcov(:,:,f,1) = (1-weight) * X(:,f,1)* X(:,f,1)';
end
for t = 2:T
    for f=1:F
        Mcov(:,:,f,t) = weight* Mcov(:,:,f,t-1) + (1-weight)*X(:,f,t)*X(:,f,t)';
    end
end

Mcov_pad = cat(4, zeros(C,C,F,1), Mcov, zeros(C,C,F,3));

P = zeros(C,C,F,T+4);
% Calculate P(k,l) in Parchami paper, using equation (4), using 2 iteration
% Estimate for special case P(k,1)

X_pad = cat(3, zeros(C,F,1), X, zeros(C,F,3));
for t = 2:T+1
    for f=1:F
        zeta(f,t) = norm(Mcov_pad(:,:,f,t)- P(:,:,f,t-1)) * inv(norm(P(:,:,f,t-1))); % it is OK to use norm of Mcov_pad(:,:,f,t) but not Mcov_pad(f,t,:,:)
        if zeta(f,t) >= TH
            zeta_norm(f,t) = 1;
        elseif (TL < zeta(f,t)) && (zeta(f,t)< TH)
            zeta_norm(f,t) = (zeta(f,t)-TL)/(TH-TL);
        else
            zeta_norm(f,t) = 0;
        end
        alpha(f,t) = alpha_min + (alpha_max - alpha_min)*zeta_norm(f,t);
        sum = w(1)*X_pad(:,f,t+1)*X_pad(:,f,t+1)' + w(2)* X_pad(:,f,t+2)*X_pad(:,f,t+2)'+ w(3)*X_pad(:,f,t+3)*X_pad(:,f,t+3)';
        P(:,:,f,t) = beta*alpha(f,t)* P(:,:,f,t-1)+(1-alpha(f,t))*X_pad(:,f,t)* X_pad(:,f,t)'+(1-beta)*alpha(f,t)*sum ;
    end
end
for j = 1:2
    for t = 2:T+1
        for f=1:F
            zeta(f,t) = norm(Mcov_pad(:,:,f,t)- P(:,:,f,t)) * inv(norm(P(:,:,f,t))); % it is OK to use norm of Mcov_pad(:,:,f,t) but not Mcov_pad(f,t,:,:)
            if zeta(f,t) >= TH
                zeta_norm(f,t) = 1;
            elseif (TL < zeta(f,t)) && (zeta(f,t)< TH)
                zeta_norm(f,t) = (zeta(f,t)-TL)/(TH-TL);
            else
                zeta_norm(f,t) = 0;
            end
            alpha(f,t) = alpha_min + (alpha_max - alpha_min)*zeta_norm(f,t);
            sum = w(1)*X_pad(:,f,t+1)*X_pad(:,f,t+1)' + w(2)* X_pad(:,f,t+2)*X_pad(:,f,t+2)'+ w(3)*X_pad(:,f,t+3)*X_pad(:,f,t+3)';
            P(:,:,f,t) = beta*alpha(f,t)* P(:,:,f,t-1)+(1-alpha(f,t))*X_pad(:,f,t)* X_pad(:,f,t)'+(1-beta)*alpha(f,t)*sum ;
        end
    end
end
Ncov = P(:,:,:,2:end-3);

% Begin Stage 2 with minima tracking, replace Ncov by past Ncov, which
% has minimum norm
Ncov_temp = Ncov; % Ncov_s2 is smoothed version of Ncov
Norm_matrix = zeros(F,T);
for f = 1:F
    for t = 1:T
        Norm_matrix(f,t) = norm(Ncov(:,:,f,t)); % notice that we can take norm of matrix in the form(:,:,f,t), but not in the form (f,t,:,:)
    end
end
for f= 1:F
    for t = M:T
        [value_min index_min] = min(Norm_matrix(f,t-M+1:t));
        Ncov_temp(:,:,f,t) = Ncov(:,:,f,t-M+index_min);
        Ncov_temp(:,:,f,t) = (1 + (M-1)/2) * Ncov_temp(:,:,f,t);
    end
end


Ncov = Ncov_temp;
for t = 1:T
    for f = 1:F
        Ncov(:,:,f,t) = 0.5 * ( Ncov(:,:,f,t) + Ncov(:,:,f,t)'); % Ensure Hermitian symmetry
        Mcov(:,:,f,t) = 0.5 * ( Mcov(:,:,f,t) + Mcov(:,:,f,t)'); % Ensure Hermitian symmetry
    end
end

Y  = zeros(F,T);
regulN = 1e-3;
minCor = 1;
eyeCoef = 1;

for t = 1:T
    for f = 1:F
        RNcov = Ncov(:,:,f,t) + regulN * diag(diag(Ncov(:,:,f,t)));
        RMcov = Mcov(:,:,f,t) + regulN * diag(diag(Mcov(:,:,f,t)));
        num = (RNcov \ RMcov - eyeCoef*eye(C));
        %lambda = real(trace(num));
        lambda = max(minCor, real(trace(num)));
        den = beta_MVDR + lambda;
        h = (num * pickMic) / den;
        Y(f,t) = h' * X(:,f,t);
        
    end
end

end
