function [x fs] = sph2wav(src_file)

% Load NIST Sphere file using the sph2pipe tool used by kaldi
%
% function [x fs] = sph2wav(src_file)

if ~exist('cleanUp', 'var') || isempty(cleanUp), cleanUp = true; end

% Set your own kaldi directory...
kaldiDir = '/home/mmandel/code/githubRO/kaldi-jsalt/tools/sph2pipe_v2.5';
dest_file = [tempname '.wav'];

[pathstr name ext] = fileparts(src_file);
cmd_str = [kaldiDir '/sph2pipe -f wav "' src_file '" "' dest_file '"'];
system(cmd_str);

[x fs] = wavread(dest_file);

delete(dest_file);
