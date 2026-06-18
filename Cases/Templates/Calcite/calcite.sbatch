#!/bin/bash
#SBATCH --chdir=/scratch/noah20/Pure
#SBATCH -p batch
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --output=calcite_%j.out
#SBATCH --error=calcite_%j.err
#SBATCH --job-name=calcite
#SBATCH --mail-user=noahmcmullan20@gmail.com
#SBATCH --mail-type=end

module purge
module load miniconda3/latest
module load gcc/11.2.0

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /scratch/noah20/PF/pflotran_env

export PETSC_DIR=/scratch/noah20/PF/src/petsc
export PETSC_ARCH=linux-gcc11-petsc3243-opt

export OMP_NUM_THREADS=1

/scratch/noah20/PF/src/petsc/linux-gcc11-petsc3243-opt/bin/mpirun \
-np $SLURM_NTASKS \
/scratch/noah20/PF/src/pflotran/src/pflotran/pflotran \
-input_prefix Calcite_diss_annotated
