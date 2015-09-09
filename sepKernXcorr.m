function [M loc C] = sepKernXcorr(LR, tau, sr, I, frame, vis, tfProd, ...
                                  refFreq, steps);

% Use the normalized kernel cross-correlation to perform
% localization and separation.  

if ~exist('vis', 'var'), vis = 0; end
if ~exist('frame', 'var'), frame = 1024; end
if ~exist('tfProd', 'var'), tfProd = 4/3; end
if ~exist('refFreq', 'var'), refFreq = 1.25; end
if ~exist('steps', 'var'), steps = 5; end

min_coh = 0.4;
crange = 0.2*sr/1000;

% for p=4:log2(size(LR,1))
%   p2=2^p; 
%   Ksame(p2:2*p2-1) = p-3; 
%   tf = [1/(refFreq*p2) refFreq*p2] * tfProd;
%   Ker{p-3} = tfKernel(tf, sr, frame); 
% end
% Ksame(1:2^5-1) = 1;
Kedge = round(ls10(1,frame/2,2*steps));
Kedge(1:ceil(steps/2)+2) = [];
Ksame = ones(1,frame/2);
for p=1:length(Kedge)-1
  Ksame(Kedge(p):Kedge(p+1)) = p;
  tf = [1/(refFreq*Kedge(p)) refFreq*Kedge(p)] * tfProd;
  Ker{p} = tfKernel(tf, sr, frame); 
end
Ksame = Ksame(1:frame/2-1);

C = normKernXcorr(LR, tau, frame, Ker, Ksame);

xcr = tau(argmax(C,3));
coh = max(C,[],3);

highc = coh > min_coh;

lochist = hist(xcr(find(highc)), tau);
loc = sort(pickPeaks(tau, lochist, I));
if vis, plot(tau, lochist), end

edges    = [min(tau) (loc(1:end-1)+loc(2:end))/2 max(tau)];
lowedge  = max(edges(1:end-1), loc-crange);
highedge = min(edges(2:end),   loc+crange);

M = zeros([size(LR,1) size(LR,2) I]);
for i=1:I
  M(:,:,i) = (xcr >= lowedge(i)) .* (xcr <= highedge(i)) .* highc;
end
