#!/bin/bash
mode_in=$1
partition_in=($2)
adj_in=$3
dist_in=$4
T0_in=$5
Tf_in=$6
sizes=($7)
max_MCS_vs_size=($8)
threads_vs_size=($9)
add_in=${10}

n_sizes=${#sizes[@]}


#for ((i = 0; i < n_sizes; i++)); do
#    if [[ "${partition_in[$i]}" == "medium" ]]; then
#        echo "medium"
#    elif [[ "${partition_in[$i]}" == "short" ]]; then
#        echo "short"
#    fi
#done

if [ -z "$add_in" ]; then
  for ((i = 0; i < n_sizes; i++)); do
      if [[ "${partition_in[$i]}" == "short" ]]; then
          sbatch --job-name="$adj_in" --time=06:00:00 --partition=short --export=ALL,mode="$mode_in",adj="$adj_in",dist="$dist_in",n=${sizes[$i]},T0="$T0_in",Tf="$Tf_in",MCS_avg=10000,max_MCS=${max_MCS_vs_size[$i]},N_config=10000,add="$add_in" med_jobfile.sh
      elif [[ "${partition_in[$i]}" == "medium" ]]; then
          sbatch --job-name="$adj_in" --time=06:00:00 --partition=medium --export=ALL,mode="$mode_in",adj="$adj_in",dist="$dist_in",n=${sizes[$i]},T0="$T0_in",Tf="$Tf_in",MCS_avg=10000,max_MCS=${max_MCS_vs_size[$i]},N_config=10000,add="$add_in" med_jobfile.sh
      fi
  done
fi