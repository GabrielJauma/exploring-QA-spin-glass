#!/bin/bash
#SBATCH --job-name=n1MW20.0
#SBATCH --workdir=.
#SBATCH --output=and_slepc-rrg-output-%j.out
#SBATCH --ntasks=2
#SBATCH --tasks-per-node=2
#SBATCH --mem=32GB
#SBATCH --time=24:00:00

cd $SLURM_SUBMIT_DIR
SCRATCHDIR=/$SLURM_SUBMIT_DIR/$SLURM_JOBID
mkdir -p -v "${SCRATCHDIR}"
### Copy files to the scratch disk of the compute node:
cp -p ./diago_rrg.py ./rrg_library.py $SCRATCHDIR
### Change working directory to scratch disk
### and do the calculation:
cd $SCRATCHDIR
source activate py38
echo $d $L $wmin $wmax $nsteps $nsamp > parameters.info
# run with parameters specify on line command:
mpirun -n $SLURM_NTASKS python ./diago_rrg.py $L $wmin $wmax $nsteps $nsamp $SLURM_JOBID
exit
