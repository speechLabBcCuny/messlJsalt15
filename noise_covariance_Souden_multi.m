function Y = noise_covariance_Souden_multi(X)

% Perform MVDR beamforming with noise covariance matrix based on [1]. And
% MVDR formula based on[2] (equation 18 with beta = 0)
%
%  Y= noise_covariance_Souden_multi(X)
%
% [1]Souden, M., Chen, J., Benesty, J., & Affes, S. (2011).
%  An integrated solution for online multichannel noise tracking and reduction.
% IEEE Transactions on Audio, Speech, and Language Processing, 19(7), 2159-2169.
% [2] Souden, M., Benesty, J., & Affes, S. (2010).
% On optimal frequency-domain multichannel linear filtering for noise reduction.
%
% Inputs:
%   X:FxTxC tensor of complex input spectrograms for each channel,
% Outputs:
%   Y: FxT matrix of estimated complex spectrum of target

[F T C] = size(X)
X = permute(X,[3,1,2]);  % now X is C F T
X = cat(3, zeros(C,F,1), X);
epsilon = 0.01;
alpha_noisy = 0.92; % follow the paper at V.Numerical Examples
alpha_v = 0.92;
alpha_p = 0.6;
L_init = 6; % Assume the first L_init there is no speech, only noise
K1 = 15;

syms x;
L = 32;
psi_0 = 18.1;
psi_tilde_0 = 28.85;
% find psi_0 by
% Hotelling_cdf = @(x)(x/L)^C * L * gamma(L)* hypergeom([C,L+1],C+1,(-x/L))/(gamma(C+1)*gamma(L-C+1)) - 0.99;
% z = fzero(Hotelling_cdf,10)
% round up psi_0 to 18.1

%  find psi_tilde_0 by
%  a = 2 * C * L;
%  B= (2*L - 2*C - 1)*(L-1)/((L-2*C-3)*(L-2*C));
%  b = 4 + (a+2)/(B-1);
%  c = a*(b-2)/(b *(L-2*C-1))
%  psi_tilde_0 = c* finv(0.99,a,b) = 28.8427
%  round up psi_tilde_0 to 28.85

Mcov = zeros(C,C,F,T+1);
Ncov = zeros(C,C,F,T+1);
Scov = zeros(C,C,F,T+1);

psi = zeros(F,T+1);
psi_tilde = zeros(F,T+1);

psi_global = zeros(F,T+1);
psi_local= zeros(F,T+1);
psi_frame = zeros(T+1);

q_frame = zeros(T+1);
q_global = zeros(F,T+1);
q_local = zeros(F,T+1);

p= zeros(F,T+1);
p1= zeros(F,T+1);
pfinal= zeros(F,T+1);

for k = 1:F
    Ncov(:,:,k,1) = 0;
    Mcov(:,:,k,1) = 0;
end

%--------------------       Iteration 1    -------------------- %
% equation (3)
for l=2:(L_init + 1)
    for k = 1:F
        p(k,l) = 0;
        Mcov(:,:,k,l) = alpha_noisy * Mcov(:,:,k,l-1) + (1 - alpha_noisy)* X(:,k,l)*X(:,k,l)';
        Ncov(:,:,k,l) = Mcov(:,:,k,l); % Assume the first L_init there is no speech, only noise
    end
end

for l = (L_init + 2):(T+1)
    for k=1:F
        Mcov(:,:,k,l) = alpha_noisy * Mcov(:,:,k,l-1) + (1 - alpha_noisy)* X(:,k,l)*X(:,k,l)';
    end
end

for l = (L_init + 2):(T+1)
    for k=1:F
        Scov(:,:,k,l) = Mcov(:,:,k,l) - Ncov(:,:,k,l-1);  %2a
        psi(k,l) = X(:,k,l)' * inv(Ncov(:,:,k,l-1)) * X(:,k,l); %2b
        psi_tilde(k,l) = trace(Ncov(:,:,k,l-1) \  Mcov(:,:,k,l)); %2c
        xi(k,l) = psi_tilde(k,l) - C; %2d
        beta(k,l) = X(:,k,l)' * inv(Ncov(:,:,k,l-1)) * Scov(:,:,k,l) * inv(Ncov(:,:,k,l-1))* Scov(:,:,k,l) * X(:,k,l);
        
        % Equation(18)
        if (psi_tilde(k,l) < C) && (psi(k,l) < psi_0)
            q_local(k,l) = 1;
        elseif (C <= psi_tilde(k,l)) && (psi_tilde(k,l) < psi_tilde_0) && (psi(k,l) < psi_0)
            q_local(k,l) = (psi_tilde_0 - psi_tilde(k,l))/(psi_tilde_0 - C);
        else
            q_local(k,l) = 0;
        end
        
    end
    % Equation 20
    for i =1:F
        psi_frame(l) =  psi(i,l) * psi_frame(l);
    end
    
    psi_frame(l) = 1/F*psi_frame(l);
    
    if (psi_frame(l) < psi_0)
        q_frame(l) = 1;
    else
        q_frame(l) = 0;
    end
    
    % Equation (19)
    % split into [1,K1], [(K1+1):(F-K1)] and (F-K1+1):F
    for k= 1:K1
        psi_global(k,l) = 1;
        for i =-K1:0
            psi_global(k,l) =  psi_global(k,l) * 0.5*(1 - cos(pi*i/K1))* psi(k-i,l);
        end
    end
    
    for k= (K1+1):(F-K1)
        psi_global(k,l) = 1;
        for i =(-K1):K1
            psi_global(k,l) =  psi_global(k,l) * 0.5*(1 - cos(pi*i/K1))* psi(k-i,l);
        end
    end
    
    for k= (F-K1+1):F
        psi_global(k,l) = 1;
        for i =0:K1
            psi_global(k,l) =  psi_global(k,l) * 0.5*(1 - cos(pi*i/K1))* psi(k-i,l);
        end
    end
    
    for k=1:F
        if (psi_global(k,l) < psi_0)
            q_global(k,l) = 1;
        else
            q_global(k,l) = 0;
        end
    end
    
    for k=1:F
        q(k,l) = q_local(k,l) * q_global(k,l) * q_frame(l);
        q(k,l) = min(q(k,l), 0.99);
        p1(k,l) = 1/(1 + q(k,l)/(1-q(k,l))*(1 + xi(k,l))*exp(-beta(k,l)/(1+xi(k,l))));
        p(k,l) = alpha_p * p(k,l-1) + (1-alpha_p)* p1(k,l);
        alpha_noise(k,l) = alpha_v + (1- alpha_v) * p(k,l);
        Ncov(:,:,k,l) = alpha_noise(k,l) * Ncov(:,:,k,l-1) + (1 - alpha_noise(k,l))*X(:,k,l)*X(:,k,l)';
    end
    
end

%--------------------       Iteration 2    -------------------- %
% Same as Iteration 1 but replace Ncov(:,:,k,l-1) by Ncov(:,:,k,l);


for l=2:(L_init + 1)
    for k = 1:F
        p(k,l) = 0;
        Mcov(:,:,k,l) = alpha_noisy * Mcov(:,:,k,l-1) + (1 - alpha_noisy)* X(:,k,l)*X(:,k,l)';
        Ncov(:,:,k,l) = Mcov(:,:,k,l); % Assume the first L_init there is no speech, only noise
    end
end

for l = (L_init + 2):(T+1)
    for k=1:F
        Mcov(:,:,k,l) = alpha_noisy * Mcov(:,:,k,l-1) + (1 - alpha_noisy)* X(:,k,l)*X(:,k,l)';
    end
end

for l = (L_init + 2):(T+1)
    for k=1:F
        Scov(:,:,k,l) = Mcov(:,:,k,l) - Ncov(:,:,k,l);  %2a
        psi(k,l) = X(:,k,l)' * inv(Ncov(:,:,k,l)) * X(:,k,l); %2b
        psi_tilde(k,l) = trace(Ncov(:,:,k,l) \  Mcov(:,:,k,l)); %2c
        xi(k,l) = psi_tilde(k,l) - C; %2d
        beta(k,l) = X(:,k,l)' * inv(Ncov(:,:,k,l)) * Scov(:,:,k,l) * inv(Ncov(:,:,k,l))* Scov(:,:,k,l) * X(:,k,l);
        
        % Equation(18)
        if (psi_tilde(k,l) < C) && (psi(k,l) < psi_0)
            q_local(k,l) = 1;
        elseif (C <= psi_tilde(k,l)) && (psi_tilde(k,l) < psi_tilde_0) && (psi(k,l) < psi_0)
            q_local(k,l) = (psi_tilde_0 - psi_tilde(k,l))/(psi_tilde_0 - C);
        else
            q_local(k,l) = 0;
        end
        
    end
    % Equation 20
    for i =1:F
        psi_frame(l) =  psi(i,l) * psi_frame(l);
    end
    
    psi_frame(l) = 1/F*psi_frame(l);
    
    if (psi_frame(l) < psi_0)
        q_frame(l) = 1;
    else
        q_frame(l) = 0;
    end
    
    % Equation (19)
    % split into [1,K1], [(K1+1):(F-K1)] and (F-K1+1):F
    for k= 1:K1
        psi_global(k,l) = 1;
        for i =-K1:0
            psi_global(k,l) =  psi_global(k,l) * 0.5*(1 - cos(pi*i/K1))* psi(k-i,l);
        end
    end
    
    for k= (K1+1):(F-K1)
        psi_global(k,l) = 1;
        for i =(-K1):K1
            psi_global(k,l) =  psi_global(k,l) * 0.5*(1 - cos(pi*i/K1))* psi(k-i,l);
        end
    end
    
    for k= (F-K1+1):F
        psi_global(k,l) = 1;
        for i =0:K1
            psi_global(k,l) =  psi_global(k,l) * 0.5*(1 - cos(pi*i/K1))* psi(k-i,l);
        end
    end
    
    for k=1:F
        if (psi_global(k,l) < psi_0)
            q_global(k,l) = 1;
        else
            q_global(k,l) = 0;
        end
    end
    
    for k=1:F
        q(k,l) = q_local(k,l) * q_global(k,l) * q_frame(l);
        q(k,l) = min(q(k,l), 0.99);
        p_final(k,l) = 1/(1 + q(k,l)/(1-q(k,l))*(1 + xi(k,l))*exp(-beta(k,l)/(1+xi(k,l))));
        
        alpha_noise(k,l) = alpha_v + (1- alpha_v) * p(k,l);
        Ncov(:,:,k,l) = alpha_noise(k,l) * Ncov(:,:,k,l-1) + (1 - alpha_noise(k,l))*X(:,k,l)*X(:,k,l)';
    end
    
end


Ncov = Ncov(:,:,:,2:end);
Mcov = Mcov(:,:,:,2:end);

for l = 1:T
    for k = 1:F
        Ncov(:,:,k,l) = 0.5 * ( Ncov(:,:,k,l) + Ncov(:,:,k,l)'); % Ensure Hermitian symmetry
        Mcov(:,:,k,l) = 0.5 * ( Mcov(:,:,k,l) + Mcov(:,:,k,l)'); % Ensure Hermitian symmetry
    end
end
Y  = zeros(F,T);
beta_MVDR = 0;
pickMic = [0;0;0;0;1;0];
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