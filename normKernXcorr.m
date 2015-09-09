function [C Dl Dr XC] = normKernXcorr(lr, tau, frame, kernel, Ksame)

% C = normKernXcorr(lr, tau, frame, kernel, Ksame)
%
% Compute a normalized cross-correlation between two signals.  Tau is
% the set of delays over which to perform the cross-correlation, frame
% is the FFT frame size, and kernel is the smoothing kernel that will
% be convolved with each signal to pool information.  Kernel can be a
% single kernel, or it can be a cell array of kernels, one per
% frequency band (there are frame/2-1 bands).  Optional argument Ksame
% indicates which kernel each frequency should use.  In this case
% kernel will be a cell array, and the kernel for frequency w will be
% kernel{Ksame(w)}.

t = tic;
[L R] = binSpec(lr, frame);
[W T] = size(L);
E = probCC(cat(3,L,R), tau, frame, 1);
Lsq = real(L).^2 + imag(L).^2;
Rsq = real(R).^2 + imag(R).^2;
XC = zeros(size(E));

if ~exist('Ksame', 'var')
  if iscell(kernel)
    Ksame = 1:length(kernel);
  else
    Ksame = ones(1,W);
  end
end
if ~iscell(kernel), kernel = {kernel}; end

if length(Ksame) ~= W
  error('Wrong length of Ksame: expected %d, found %d', W, length(Ksame));
end

uk = unique(Ksame);

convFn = @fftfilt2;
%convFn = @conv2;
if length(uk) < W/5
  % Do a smaller number of 2D convs with the whole spectrograms,
  % then pick out the right rows for each kernel
  for k=1:length(uk)
    wk = find(Ksame == uk(k));
    K = kernel{uk(k)};
    kW = size(K,1);
    
    for i=1:size(E,3)
      XCtmp = convFn(E(:,:,i), K, 'same');
      XC(wk,:,i) = XCtmp(wk,:);
    end
    
    Dtmp = convFn(Lsq, K, 'same');
    Dl(wk,:) = Dtmp(wk,:);
    Dtmp = convFn(Rsq, K, 'same');
    Dr(wk,:) = Dtmp(wk,:);
    %fprintf('.')
  end
else
  % Probably faster just to do a lot of small 2D convs
  for w=1:W
    K = kernel{w};
    kW = size(K,1);
    wmin = max(w-ceil(kW/2)+1, 1);
    wmax = min(w+floor(kW/2), W);
    widx = wmin:wmax;
    wout = find(widx == w);
    
    for i=1:size(E,3)
      XCtmp = convFn(E(widx,:,i), K, 'same');
      XC(w,:,i) = XCtmp(wout,:);
    end
    
    Dtmp = convFn(Lsq(widx,:), K, 'same');
    Dl(w,:) = Dtmp(wout,:);
    Dtmp = convFn(Rsq(widx,:), K, 'same');
    Dr(w,:) = Dtmp(wout,:);
    %if mod(w,10) == 0, fprintf('.'), end
  end
end
%fprintf('\n');

denom = sqrt(Dl.*Dr);
%XC = real(XC);
C = real(XC) ./ repmat(denom, [1 1 size(E,3)]);
toc(t);
