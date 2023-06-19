#!/bin/bash
#SBATCH -o jobs_out_and_err/job%J.out
#SBATCH -e jobs_out_and_err/job%J.err
#SBATCH --ntasks=10
# #SBATCH --tasks-per-node=5
#SBATCH --mem=10GB
module load Miniconda3/4.9.2
source /dragofs/sw/foss/0.2/software/Miniconda3/4.9.2/etc/profile.d/conda.sh
conda activate Architecture_v1
if [ "$mode" = "binned" ]; then
  mpirun -n $SLURM_NTASKS python3.8 spin_glass_analysis_cluster_drago.py $adj $dist $n $T0 $Tf $MCS_avg $max_MCS $N_config $SLURM_JOBID $add
elif [ "$mode" = "fast" ]; then
  mpirun -n $SLURM_NTASKS python3.8 spin_glass_analysis_cluster_fast_drago.py $adj $dist $n $T0 $Tf $max_MCS $N_config $SLURM_JOBID $add
fi
exit
