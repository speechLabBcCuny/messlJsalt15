function dicts = analyzeKaldiRes(resDir, nPrint, diffDicts)

% Analyze the results of a kaldi experiment in more depth than just WER

if ~exist('nPrint', 'var') || isempty(nPrint), nPrint = 10; end
if ~exist('diffDicts', 'var'), diffDicts = []; end

dictNames = {'ref','cor','ins','del','sub','subIn','subOut'};
dictLongNames = {'Appearances', 'Correct', 'Insertions', 'Deletions', 'Substitutions', 'Substituted in', 'Substituted out'};
ratioDicts = [2 3 4 6 7];

% TODO: make this work for actual kaldi directory structure
referenceFile = fullfile(resDir, 'test_filt.txt');
transcriptFile = fullfile(resDir, '11.txt');

[refFiles refWords] = loadTranscripts(referenceFile);
[traFiles traWords] = loadTranscripts(transcriptFile);

% Match sentences
[files refI traI] = intersect(refFiles, traFiles);
refWords = refWords(refI);
traWords = traWords(traI);

for d = 1:length(dictNames)
    dicts.(dictNames{d}) = struct('wA',  0);
end

% run dynamic programming on each sentence
for f = 1:length(files)
    ref = refWords{f};
    tra = traWords{f};
    [labels refAli traAli] = alignSeqs(ref, tra);
    
    % Stats for WER
    numWords(f) = length(ref);
    numErr(f) = sum(labels ~= 0);
    numCor(f) = sum(labels == 0);
    numIns(f) = sum(labels == 1);
    numDel(f) = sum(labels == 2);
    numSub(f) = sum(labels == 3);

    for w = 1:length(ref)
        dicts.ref = dictInc(dicts.ref, ref{w});
    end
    
    for w = 1:length(refAli)
        if labels(w) == 0
            dicts.cor = dictInc(dicts.cor, traAli{w});
        elseif labels(w) == 1
            dicts.ins = dictInc(dicts.ins, traAli{w});
        elseif labels(w) == 2
            dicts.del = dictInc(dicts.del, refAli{w});
        elseif labels(w) == 3
            dicts.subOut = dictInc(dicts.subOut, refAli{w});
            dicts.subIn = dictInc(dicts.subIn,  traAli{w});
            dicts.sub = dictInc(dicts.sub, sprintf('%s_to_%s', refAli{w}, traAli{w}));
        else
            error('Bad label: %d', labels(w))
        end
    end
end

% Plots histograms of counts
figure(1)
bins = 0:max([numWords numErr]);
plotHists(bins, numErr, numCor, numIns, numDel, numSub, numWords);

% Plot histograms of ratios
figure(2)
bins = linspace(0, max(numErr ./ numWords), 30);
plotHists(bins, numErr./numWords, numCor./numWords, numIns./numWords, numDel./numWords, numSub./numWords);


% TODO: convert numeric transcripts to text

% TODO: simulate the global I/S/D rates across all sentences
%[simCor simIns simDel simSub] = shuffleErrors(numWords, numCor, numIns, numDel, numSub);

% plot histograms of those to compare

% Print most frequent words for each list
printHeading('Counts')
for d = 1:length(dictNames)
    printTopEntries(dicts.(dictNames{d}), nPrint, dictLongNames{d});
end

% print top proportional words of each type
printHeading('Proportions')
for d = ratioDicts
    ratio = combineStructs(@(x,y) (x+1) ./ (y+1), dicts.(dictNames{d}), dicts.ref);
    printTopEntries(ratio, nPrint, dictLongNames{d});
end

% Compare to results from another system
if ~isempty(diffDicts)
    printHeading('Differences')
    for d = 1:length(dictNames)
        diff = combineStructs(@minus, dicts.(dictNames{d}), diffDicts.(dictNames{d}));
        printTopEntries(diff, nPrint, dictLongNames{d});
    end
end


function [files words] = loadTranscripts(file)
lines = textArray(file);
sents = listMap(@(x) split(chop(x), ' '), sort(lines(1:end-1)));
files = listMap(@(x) x{1}, sents);
words = listMap(@(x) x(2:end), sents);
words = listMap(@(xs) regexprep(xs, '\W', '_'), words);

function printSent(words, widths, reps)
for i = 1:length(words)
    if reps(i)
        fprintf('%s ', repmat('-', 1, widths{i}))
    else
        fprintf('%s%s ', words{i}, repmat(' ', 1, widths{i}-length(words{i})))
    end
end
fprintf('\n')

function plotHists(bins, numErr, numCor, numIns, numSub, numDel, numWords)
if nargin < 7, numWords = []; end

% Plots histograms of counts
h(1,:) = hist(numErr, bins);
h(2,:) = hist(numCor, bins);
h(3,:) = hist(numIns, bins);
h(4,:) = hist(numDel, bins);
h(5,:) = hist(numSub, bins);
labels = {'Errors', 'Correct', 'Insertions', 'Deletions', 'Substitutions'};
if ~isempty(numWords)
    h(6,:) = hist(numWords, bins);
    labels{end+1} = 'Words';
end

plot(bins, h', 'LineWidth', 2)
legend(labels)

% function [simCor simIns simDel simSub] = shuffleErrors(numWords, numCor, numIns, numDel, numSub)

function d = dictInc(d, word)
% Word should not have any forbidden characters at this point...
word = ['w' word];
if ~isfield(d, word)
    d.(word) = 0;
end
d.(word) = d.(word) + 1;

function [k v totalCount] = topEntries(d, n)
% Take the top n key-value pairs from structure d, where top is defined as
% having the highest value (which should be a number).
keys = fieldnames(d);
vals = cell2mat(struct2cell(d));
for i=1:length(keys)
    srtKeys{i} = sprintf('%06d%s', vals(i), keys{i});
end
[~,ord] = sort(vals, 'descend');
k = keys(ord(1:n));
v = vals(ord(1:n));
totalCount = sum(vals);

function printTopEntries(d, n, title)
[k v totalCount] = topEntries(d, n);
fprintf('\n%s: %g\n', title, totalCount);
for i = 1:length(k)
    fprintf('%s: %g\n', k{i}(2:end), v(i));
end

function z = combineStructs(fn, x, y)
% Create a structure with the keys of x and y and the values combined using
% the function zv = fn(xv, yv);  If a key only exists in one structure, the
% value of 0 will be used in its place.
keys = union(fieldnames(x), fieldnames(y));
for k = 1:length(keys)
    key = keys{k};
    if ~isfield(x, key)
        x.(key) = 0;
    end
    if ~isfield(y, key)
        y.(key) = 0;
    end
    z.(key) = fn(x.(key), y.(key));
end

function printHeading(title)
fprintf('\n\n==========================================\n')
fprintf('%s\n', title)
fprintf('==========================================\n')
