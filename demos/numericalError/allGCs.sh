#!/bin/sh
#SBATCH --output=comptime.out
#SBATCH --error=comptime.err
#SBATCH --job-name=comptime
#SBATCH --partition=medium
#SBATCH --time=1440
#SBATCH --array=[0-165]
#SBATCH --cpus-per-task=6 

# this job launchs the same jobs for each cluster
# it then uses 6 cpu's to parallelize different parameters per cluster

GCname=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" GCnames.txt)

srun python stream_time_reversability.py "$GCname"