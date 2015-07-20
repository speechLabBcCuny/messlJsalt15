#!/bin/bash -ue

# Run an arbitrary matlab command using qsub.  These is no protection
# against ill-formed or evil matlab commands, so watch out!  There is
# protection against exceptions.  They will be printed to the console
# and matlab and the job will exit.
#
# run with commandline:
#   qsub [options] ./qsubMatlab.sh [matlabCommand]
#
# e.g.
#   qsub ./qsubMatlab.sh "fprintf('\$SLURM_PROCID of \$SLURM_NTASKS\n')"

MAT_DIR=$HOME/code/matlab
CUR_DIR=`pwd`
# CHIME_CODE_DIR=/data/corpora/chime3/CHiME3/tools/

echo "$0 \"$@\""
# matlab -singleCompThread -nodisplay -r "try; cd $MAT_DIR ; startup; cd $CUR_DIR ; addpath(genpath('$CHIME_CODE_DIR')); $@; catch ex; disp(getReport(ex)); end; quit"
matlab -singleCompThread -nodisplay -r "try; cd $MAT_DIR ; startup; cd $CUR_DIR ; warning('off', 'MATLAB:audiovideo:wavread:functionToBeRemoved'); $@; catch ex; disp(getReport(ex)); end; quit"
