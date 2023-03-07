#!/bin/sh
#SBATCH --account=pinaki.sarder
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000mb
#SBATCH --partition=gpu
#SBATCH --gpus=geforce:2
#SBATCH --time=72:00:00
#SBATCH --output=hail.out
#SBATCH --job-name="ObjExt"
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST="$SLURM_JOB_NODELIST
echo "SLURM_NNODES="$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR

echo "working directory = "$SLURM_SUBMIT_DIR
ulimit -s unlimited
module load python/3.6.5
module load tensorflow/1.14.0
module use /home/nlucarelli/privatemodules
module load openslide/3.4.0
module load joblib/0.11
module load imgaug/0.4.0
module load imageio/2.3.0
module list
which python

echo "Launch job"
#python3 xml_to_mask_KPMP.py
python3 Kfold_RNN.py
#
echo "All Done!"
