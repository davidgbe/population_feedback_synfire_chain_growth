#!/bin/bash

## Presets
#SBATCH --job-name=syn_param

#SBATCH -p stf
#SBATCH -A stf

#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --mem=20G

#SBATCH --chdir=.

## Module import
module load contrib/python_3.8/3.8
CURRDIR=$(pwd)
echo $CURRDIR
## Commands to run
cd //gscratch/stf/davidgbe/compneuro
source davidgbe-env/bin/activate
cd $CURRDIR
python run.py --title sweep_3_w_e_i_7e-5 --index INDEX --rng_seed SEED --dropout_per 0 --w_ee W_E_E --w_ei W_E_I --w_ie W_I_E
wait
deactivate

## Exit
exit 0
