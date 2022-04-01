#!/bin/bash

caffeinate python3 run.py --title growth_test_3 --alpha 0.5 --beta 0.015 --gamma 0 --fr_single_line_attr 0 --rng_seed $1 --dropout_per 0
# beta 5e-1 2e-1