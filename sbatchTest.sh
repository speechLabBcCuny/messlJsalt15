#!/bin/bash
#
#SBATCH --job-name=test
#SBATCH --output=slurmTestRes.txt
#
#SBATCH --ntasks=4

echo "$SLURM_PROCID"
# srun printenv
# srun printenv SLURM_PROCID
# srun echo "$SLURM_PROCID"
# srun echo "$SLURM_PROCID of $SLURM_NTASKS"
# srun matlab -nodisplay -r "try; fprintf('$SLURM_PROCID of $SLURM_NTASKS\n'); catch ex; fprintf('ERROR:\n'); disp(ex); end; quit"
