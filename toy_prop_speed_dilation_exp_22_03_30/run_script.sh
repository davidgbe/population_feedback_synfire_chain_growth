#!/bin/bash

caffeinate python3 run.py --title 5_per_no_tri --alpha 0 --beta 0 --gamma 0 --fr_single_line_attr 0 --rng_seed $1 --dropout_per $2
# beta 5e-1 2e-1