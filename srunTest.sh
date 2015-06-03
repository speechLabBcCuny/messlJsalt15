#!/bin/bash

# run with commandline:
# srun -n 4 -o slurmTestRes_%j-%2t.txt ./slurmTest.sh

# echo "$SLURM_PROCID"
# printenv
# printenv SLURM_PROCID
# echo "$SLURM_PROCID"
# echo "$SLURM_PROCID of $SLURM_NTASKS"
matlab -nodisplay -r "try; fprintf('$SLURM_PROCID of $SLURM_NTASKS\n'); catch ex; fprintf('ERROR:\n'); disp(ex); end; quit"
