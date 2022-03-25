#!/bin/bash

caffeinate python3 run.py --title stasis_test --alpha 0.5 --beta 0.01 --gamma 0.1 --fr_single_line_attr 0 --rng_seed $1 --dropout_per $3 --load_run $2