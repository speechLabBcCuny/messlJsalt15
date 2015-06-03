#!/bin/bash -ue

# Run an arbitrary matlab command using slurm.  These is no protection
# against ill-formed or evil matlab commands, so watch out!  There is
# protection against exceptions.  They will be printed to the console
# and matlab and the job will exit.
#
# run with commandline:
#   srun [options] ./srunMatlab.sh [matlabCommand]
#
# e.g.
#   srun -n 2 ./srunMatlab.sh "fprintf('\$SLURM_PROCID of \$SLURM_NTASKS\n')"

echo "$0 \"$@\""
matlab -nodisplay -r "try; $@; catch ex; disp(getReport(ex)); end; quit"
