function mOut = mungeMaskForMvdr(mIn, minOnThresh, minOffThresh)

% Munge a separation mask to work well for MVDR noise estimation control

if ~exist('minOnThresh', 'var') || isempty(minOnThresh), minOnThresh = 0.35; end
if ~exist('minOffThresh', 'var') || isempty(minOffThresh), minOffThresh = 0; end

fracOff = mean(1-mIn, 1);
fracOn  = mean(  mIn, 1);

% If at least minOnThresh fraction of points are on at a given time, then
% all points at that time frame should be on. If less than minOffThresh
% fraction of points are off, then all points at that time frame should be
% off.

mOut = bsxfun(@min, bsxfun(@max, mIn, fracOn > minOnThresh), fracOff > minOffThresh);
