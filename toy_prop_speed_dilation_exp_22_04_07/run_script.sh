#!/bin/bash

caffeinate python3 run.py --title sweep_2_w_e_i_7e-5 --index 0 --rng_seed $1 --dropout_per 0 --w_ee 2e-3 --w_ei 7e-5 --w_ie 7.5e-5