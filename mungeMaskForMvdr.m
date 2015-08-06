function mOut = mungeMaskForMvdr(mIn, minOffPts, minOnThresh, minOffThresh)

% Munge a separation mask to work well for MVDR noise estimation control

if ~exist('minOffPts', 'var') || isempty(minOffPts), minOffPts = 0; end
if ~exist('minOnThresh', 'var') || isempty(minOnThresh), minOnThresh = 0.35; end
if ~exist('minOffThresh', 'var') || isempty(minOffThresh), minOffThresh = 0; end

% Turn off mask at beginning and end (twice as long at the end) to get at
% least minOffPts points where the MVDR mask is off to avoid NaNs in
% spatial covariance.
mIn(:,1:floor(minOffPts/3)) = 0;
mIn(:,end-ceil(minOffPts*2/3)+1:end) = 0;

fracOff = mean(1-mIn, 1);
fracOn  = mean(  mIn, 1);

% If at least minOnThresh fraction of points are on at a given time, then
% all points at that time frame should be on. If less than minOffThresh
% fraction of points are off, then all points at that time frame should be
% off.

mOut = bsxfun(@min, bsxfun(@max, mIn, fracOn > minOnThresh), fracOff > minOffThresh);
