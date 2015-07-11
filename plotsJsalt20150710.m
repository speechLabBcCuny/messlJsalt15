function plotsJsalt20150710(toDisk, startAt)

if ~exist('toDisk', 'var') || isempty(toDisk), toDisk = false; end
if ~exist('startAt', 'var') || isempty(startAt), startAt = 0; end

inDir = '~/data/jsalt20150710';
outDir = '~/data/plots/jsalt20150710';
files = {};
cax = [-80 10];

plotSpecsCopyWavs(inDir, outDir, toDisk, startAt, files, cax, [], [0 5]);
