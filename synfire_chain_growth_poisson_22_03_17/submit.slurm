#!/bin/bash

## Presets
#SBATCH --job-name=growth

#SBATCH -p stf
#SBATCH -A stf

#SBATCH --nodes=1
#SBATCH --time=16:00:00
#SBATCH --ntasks=10
#SBATCH --mem=100G

#SBATCH --chdir=.

## Module import
module load contrib/python_3.8/3.8
CURRDIR=$(pwd)
echo $CURRDIR
## Commands to run
cd //gscratch/stf/davidgbe/compneuro
source davidgbe-env/bin/activate
cd $CURRDIR
python run.py --title growth_3 --alpha 0.5 --beta 1 --gamma 0 --fr_single_line_attr 0 --rng_seed SEED --dropout_per 0 &
wait
deactivate

## Exit
exit 0
