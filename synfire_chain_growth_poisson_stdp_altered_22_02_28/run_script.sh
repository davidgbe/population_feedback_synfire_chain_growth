#!/bin/bash

caffeinate python3 run.py --title growth_test_2 --alpha 6e-2 --beta 2e-3 --gamma 0 --fr_single_line_attr 0 --rng_seed $1 --dropout_per 0
# beta 5e-1 2e-1