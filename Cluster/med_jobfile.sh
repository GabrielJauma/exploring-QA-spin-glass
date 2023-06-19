#!/bin/bash
#SBATCH -o jobs_out_and_err/job%J.out
#SBATCH -e jobs_out_and_err/job%J.err
#SBATCH --ntasks=10
# #SBATCH --tasks-per-node=5
#SBATCH --mem=10GB
module load miniconda3/4.11.0
conda activate Architecture_v1
if [ "$mode" = "binned" ]; then
  mpirun -n $SLURM_NTASKS python3.8 spin_glass_analysis_cluster_binned.py $adj $dist $n $T0 $Tf $MCS_avg $max_MCS $N_config $SLURM_JOBID $add
elif [ "$mode" = "fast" ]; then
  mpirun -n $SLURM_NTASKS python3.8 spin_glass_analysis_cluster_fast.py $adj $dist $n $T0 $Tf $max_MCS $N_config $SLURM_JOBID $add
fi
exit
