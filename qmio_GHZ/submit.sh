#!/usr/bin/env sh

#SBATCH -p qpu
#SBATCH --mem 16G
#SBATCH -t 02:00:00

module load qmio/hpc gcccore/12.3.0 python/3.11.9
module load qmio/hpc gcccore/12.3.0 numpy/2.1.2-python-3.11.9
module load qmio/hpc gcccore/12.3.0 matplotlib/3.6.3-python-3.11.9
module load qmio/hpc gcc/12.3.0 qiskit/2.2.3-python-3.11.9
module load qmio-run
module load qmio-tools

python main.py run ".\experiment_definitions\real_qmio.yml"