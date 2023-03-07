#!/bin/sh
#SBATCH --account=pinaki.sarder
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000mb
#SBATCH --partition=gpu
#SBATCH --gpus=geforce
#SBATCH --time=72:00:00
#SBATCH --output=fcnn.out
#SBATCH --job-name="FCNN"
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST="$SLURM_JOB_NODELIST
echo "SLURM_NNODES="$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR

echo "working directory = "$SLURM_SUBMIT_DIR
ulimit -s unlimited
module load tensorflow/2.6.0
module list
which python

echo "Launch job"
python model_ESRD_agg.py
#
echo "All Done!"
