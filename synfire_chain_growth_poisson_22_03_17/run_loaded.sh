#!/bin/bash

caffeinate python3 run.py --title stasis_test --alpha 0.5 --beta 1 --gamma 1e-1 --fr_single_line_attr 0 --rng_seed $1 --dropout_per 0 --load_run $2
# beta 5e-1 2e-1