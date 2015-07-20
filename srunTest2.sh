#!/bin/bash

# run with commandline:
# srun -n 4 -o slurmTestRes_%j-%2t.txt ./slurmTest2.sh

ARGSTR="$@"  # single space-separated string of arguments
ARGS=$(printf ",'%s'" $ARGSTR)  # comma and quote-separated list of arguments
ARGS=${ARGS:1}

# echo "$@"
# echo "$SLURM_PROCID"
# printenv
# printenv SLURM_PROCID
# echo "$SLURM_PROCID"
# echo "$SLURM_PROCID of $SLURM_NTASKS. Args: $ARGS"
matlab -nodisplay -r "try; fprintf('$SLURM_PROCID of $SLURM_NTASKS. Args: $ARGSTR\n'); catch ex; fprintf('ERROR:\n'); disp(ex); end; quit"
