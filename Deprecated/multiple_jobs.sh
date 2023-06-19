#for run in {1..2}; do sbatch --export=ALL,adj=zephyr,dist=gaussian_EA,n=100,T0=0.5,Tf=5.0,MCS_avg=10000,max_MCS=80000,N_config=10000 med_jobfile.sh; done
#for run in {1..4}; do sbatch --export=ALL,adj=zephyr,dist=gaussian_EA,n=200,T0=0.5,Tf=5.0,MCS_avg=10000,max_MCS=160000,N_config=10000 med_jobfile.sh; done
#for run in {1..16}; do sbatch --export=ALL,adj=zephyr,dist=gaussian_EA,n=400,T0=0.5,Tf=5.0,MCS_avg=10000,max_MCS=320000,N_config=10000 med_jobfile.sh; done
#for run in {1..32}; do sbatch --export=ALL,adj=zephyr,dist=gaussian_EA,n=800,T0=0.5,Tf=5.0,MCS_avg=10000,max_MCS=640000,N_config=10000 med_jobfile.sh; done
#for run in {1..64}; do sbatch --export=ALL,adj=zephyr,dist=gaussian_EA,n=1600,T0=0.5,Tf=5.0,MCS_avg=10000,max_MCS=1280000,N_config=10000 med_jobfile.sh; done
#for run in {1..64}; do sbatch --export=ALL,adj=zephyr,dist=gaussian_EA,n=3200,T0=0.5,Tf=2.5,MCS_avg=10000,max_MCS=640000,N_config=10000 med_jobfile.sh; done
#for run in {1..2}; do sbatch --export=ALL,adj=1D+,dist=gaussian_EA,n=100,T0=1.0,Tf=2.5,MCS_avg=10000,max_MCS=80001,N_config=10000,add=5.0 med_jobfile.sh; done
#for run in {1..4}; do sbatch --export=ALL,adj=1D+,dist=gaussian_EA,n=200,T0=1.0,Tf=2.5,MCS_avg=10000,max_MCS=80001,N_config=10000,add=5.0 med_jobfile.sh; done
#for run in {1..16}; do sbatch --export=ALL,adj=1D+,dist=gaussian_EA,n=400,T0=1.0,Tf=2.5,MCS_avg=10000,max_MCS=160001,N_config=10000,add=5.0 med_jobfile.sh; done
#for run in {1..32}; do sbatch --export=ALL,adj=1D+,dist=gaussian_EA,n=800,T0=1.0,Tf=2.5,MCS_avg=10000,max_MCS=320001,N_config=10000,add=5.0 med_jobfile.sh; done
#for run in {1..64}; do sbatch --export=ALL,adj=1D+,dist=gaussian_EA,n=1600,T0=1.0,Tf=2.5,MCS_avg=10000,max_MCS=640001,N_config=10000,add=5.0 med_jobfile.sh; done
for run in {1..2}; do sbatch --export=ALL,adj=random_regular_9,dist=gaussian_EA,n=100,T0=1.0,Tf=4.0,MCS_avg=10000,max_MCS=80000,N_config=10000 med_jobfile.sh; done
for run in {1..4}; do sbatch --export=ALL,adj=random_regular_9,dist=gaussian_EA,n=200,T0=1.0,Tf=4.0,MCS_avg=10000,max_MCS=80000,N_config=10000 med_jobfile.sh; done
for run in {1..16}; do sbatch --export=ALL,adj=random_regular_9,dist=gaussian_EA,n=400,T0=1.0,Tf=4.0,MCS_avg=10000,max_MCS=160000,N_config=10000 med_jobfile.sh; done
for run in {1..32}; do sbatch --export=ALL,adj=random_regular_9,dist=gaussian_EA,n=800,T0=1.0,Tf=4.0,MCS_avg=10000,max_MCS=640000,N_config=10000 med_jobfile.sh; done
for run in {1..64}; do sbatch --export=ALL,adj=random_regular_9,dist=gaussian_EA,n=1600,T0=1.0,Tf=4.05,MCS_avg=10000,max_MCS=640000,N_config=10000 med_jobfile.sh; done

