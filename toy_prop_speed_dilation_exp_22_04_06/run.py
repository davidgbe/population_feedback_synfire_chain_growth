from copy import deepcopy as copy
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd
import pickle
from collections import OrderedDict
import os
from scipy.ndimage.interpolation import shift
import scipy.io as sio
from scipy.optimize import curve_fit
from scipy.sparse import csc_matrix, csr_matrix, kron
from functools import reduce, partial
import argparse
import time
import tracemalloc

from aux import *
from disp import *
from ntwk import LIFNtwkG
from utils.general import *
from utils.file_io import *

cc = np.concatenate

parser = argparse.ArgumentParser()
parser.add_argument('--title', metavar='T', type=str, nargs=1)
parser.add_argument('--alpha', metavar='a', type=float, nargs=1)
parser.add_argument('--beta', metavar='b', type=float, nargs=1)
parser.add_argument('--gamma', metavar='c', type=float, nargs=1)
parser.add_argument('--fr_single_line_attr', metavar='s', type=int, nargs=1)
parser.add_argument('--rng_seed', metavar='r', type=int, nargs=1)
parser.add_argument('--load_run', metavar='l', type=str, nargs=1)
parser.add_argument('--dropout_per', metavar='d', type=float, nargs=1)


args = parser.parse_args()

print(args)

# PARAMS
## NEURON AND NETWORK MODEL
M = Generic(
    # Excitatory membrane
    C_M_E=1e-6,  # membrane capacitance
    G_L_E=0.1e-3,  # membrane leak conductance (T_M (s) = C_M (F/cm^2) / G_L (S/cm^2))
    E_L_E=-.07,  # membrane leak potential (V)
    V_TH_E=-.043,  # membrane spike threshold (V)
    T_R_E=1e-3,  # refractory period (s)
    E_R_E=-0.065, # reset voltage (V)
    
    # Inhibitory membrane
    C_M_I=1e-6,
    G_L_I=.4e-3, 
    E_L_I=-.057,
    V_TH_I=-.043,
    T_R_I=0, #0.25e-3,
    E_R_I=-.055, # reset voltage (V)
    
    # syn rev potentials and decay times
    E_E=0, E_I=-.09, E_A=-.07, T_E=.004, T_I=.004, T_A=.006,
    
    N_EXC=1000,
    N_INH=200,
    M=20,
    
    # Input params
    DRIVING_HZ=2, # 2 Hz lambda Poisson input to system
    N_DRIVING_CELLS=10,
    PROJECTION_NUM=10,
    INPUT_STD=1e-3,
    BURST_T=1.5e-3,
    INPUT_DELAY=50e-3,
    
    # OTHER INPUTS
    SGM_N=10e-10,  # noise level (A*sqrt(s))
    I_EXT_B=0,  # additional baseline current input

    # Connection probabilities
    CON_PROB_R=0.,
    E_I_CON_PROB=0.01,
    I_E_CON_PROB=1.,

    # Weights
    W_E_I_R=3e-5,
    W_I_E_R=3e-5,
    W_A=0,
    W_E_E_R=0.26 * 0.004 * 0.6,
    W_E_E_R_MIN=1e-6,
    W_E_E_R_MAX=0.26 * 0.004 * 1.5, #1.5, then 1, then 0.2;
    CELL_OUTPUT_MAX=0.26 * 0.004 * 1.3 * 3,

    # Dropout params
    DROPOUT_MIN_IDX=0,
    DROPOUT_MAX_IDX=0, # set elsewhere
    DROPOUT_ITER=2,
    DROPOUT_SEV=args.dropout_per[0],

    POP_FR_TRIALS=(100, 110),

    # Synaptic plasticity params
    TAU_STDP_PAIR_EE=15e-3,
    TAU_STDP_PAIR_EI=2e-3,

    SINGLE_CELL_FR_SETPOINT_MIN=6,
    SINGLE_CELL_FR_SETPOINT_MIN_STD=2,
    SINGLE_CELL_LINE_ATTR=bool(args.fr_single_line_attr[0]),
    ETA=0.15,
    ALPHA=args.alpha[0], #3e-2
    BETA=args.beta[0], #1e-3,
    GAMMA=args.gamma[0], #1e-4,
)

S = Generic(RNG_SEED=args.rng_seed[0], DT=0.22e-3, T=500e-3, EPOCHS=1)
np.random.seed(S.RNG_SEED)

M.N_UVA = 0

M.W_U_E = 0.26 * 0.004

M.TAU_PAIR_EE_CENTER = int(4.4e-3 / S.DT) + 1
M.CUT_IDX_TAU_PAIR_EE = int(3 * M.TAU_STDP_PAIR_EE / S.DT)
kernel_base_ee = np.arange(2 * M.CUT_IDX_TAU_PAIR_EE + 1) - M.CUT_IDX_TAU_PAIR_EE - M.TAU_PAIR_EE_CENTER
M.KERNEL_PAIR_EE = np.exp(-1 * np.abs(kernel_base_ee) * S.DT / M.TAU_STDP_PAIR_EE).astype(float)
M.KERNEL_PAIR_EE = np.where(kernel_base_ee > 0, 1, -1) * M.KERNEL_PAIR_EE

M.CUT_IDX_TAU_PAIR_EI = int(2 * M.TAU_STDP_PAIR_EI / S.DT)
kernel_base_ei = np.arange(2 * M.CUT_IDX_TAU_PAIR_EI + 1) - M.CUT_IDX_TAU_PAIR_EI
M.KERNEL_PAIR_EI = np.exp(-1 * np.abs(kernel_base_ei) * S.DT / M.TAU_STDP_PAIR_EI).astype(float)
M.KERNEL_PAIR_EI[M.CUT_IDX_TAU_PAIR_EI:] *= -1
M.KERNEL_PAIR_EI *= 0

M.DROPOUT_MAX_IDX = M.N_EXC

## SMLN

print('T_M_E =', 1000*M.C_M_E/M.G_L_E, 'ms')  # E cell membrane time constant (C_m/g_m)

def ff_unit_func(m):
    w = m.W_E_E_R / m.PROJECTION_NUM
    return gaussian_if_under_val(1, (m.PROJECTION_NUM, m.PROJECTION_NUM), w, 0.2 * w)

def generate_ff_chain(size, unit_size, unit_funcs, ff_deg=[0, 1], tempering=[1., 1.]):
    if size % unit_size != 0:
        raise ValueError('unit_size does not evenly divide size')

    if len(ff_deg) != len(tempering):
        raise ValueError('ff_deg and tempering must be the same length')

    n_units = int(size / unit_size)
    chain_order = np.arange(0, n_units, dtype=int)
    mat = np.zeros((size, size))

    for idx in chain_order:
        layer_start = unit_size * idx
        for j, ff_idx in enumerate(ff_deg):
            if idx + ff_idx < len(chain_order) and idx + ff_idx >= 0:
                ff_layer_idx = chain_order[idx + ff_idx]
                ff_layer_start = unit_size * ff_layer_idx

                mat[ff_layer_start : ff_layer_start + unit_size, layer_start : layer_start + unit_size] = unit_funcs[j](m) * tempering[j]
    return mat

### RUN_TEST function

def run_test(m, output_dir_name, n_show_only=None, add_noise=True, dropout={'E': 0, 'I': 0},
    w_r_e=None, w_r_i=None, epochs=500, e_cell_pop_fr_setpoint=None):

    output_dir = f'./figures/{output_dir_name}'
    os.makedirs(output_dir)

    robustness_output_dir = f'./robustness/{output_dir_name}'
    os.makedirs(robustness_output_dir)

    sampled_cell_output_dir = f'./sampled_cell_rasters/{output_dir_name}'
    os.makedirs(sampled_cell_output_dir)
    
    w_u_proj = np.diag(np.ones(m.N_DRIVING_CELLS)) * m.W_U_E * 0.5
    w_u_uva = np.diag(np.ones(m.N_UVA)) * m.W_U_E * 1.5 * 0

    w_u_e = np.zeros([m.N_EXC + m.N_UVA, m.N_DRIVING_CELLS + m.N_UVA])
    w_u_e[:m.N_DRIVING_CELLS, :m.N_DRIVING_CELLS] += w_u_proj
    w_u_e[m.N_EXC:(m.N_EXC + m.N_UVA), m.N_DRIVING_CELLS:(m.N_DRIVING_CELLS + m.N_UVA)] += w_u_uva

    ## input weights
    w_u = {
        # localized inputs to trigger activation from start of chain
        'E': np.block([
            [ w_u_e ],
            [ np.zeros([m.N_INH, m.N_DRIVING_CELLS + m.N_UVA]) ],
        ]),

        'I': np.zeros((m.N_EXC + m.N_UVA + m.N_INH, m.N_DRIVING_CELLS + m.N_UVA)),

        'A': np.zeros((m.N_EXC + m.N_UVA + m.N_INH, m.N_DRIVING_CELLS + m.N_UVA)),
    }

    def unit_func():
        return ff_unit_func(m)

    if w_r_e is None:
        w_e_e_r = generate_ff_chain(m.N_EXC, m.PROJECTION_NUM, [unit_func] * 1, ff_deg=np.arange(1) + 1, tempering=np.exp(-4/15 * np.arange(1)))
        np.fill_diagonal(w_e_e_r, 0.)

        e_i_r = gaussian_if_under_val(m.E_I_CON_PROB, (m.N_INH, m.N_EXC), m.W_E_I_R, 0)

        w_r_e = np.block([
            [ w_e_e_r, np.zeros((m.N_EXC, m.N_INH)) ],
            [ e_i_r,  np.zeros((m.N_INH, m.N_INH)) ],
        ])

    if w_r_i is None:

        i_e_r = gaussian_if_under_val(m.I_E_CON_PROB, (m.N_EXC, m.N_INH), 0.5 * m.W_I_E_R, 0)

        w_r_i = np.block([
            [ np.zeros((m.N_EXC, m.N_EXC + m.N_UVA)), i_e_r],
            [ np.zeros((m.N_UVA + m.N_INH, m.N_EXC + m.N_UVA + m.N_INH)) ],
        ])
    
    ## recurrent weights
    w_r = {
        'E': w_r_e,
        'I': w_r_i,
        'A': np.block([
            [ m.W_A * np.diag(np.ones((m.N_EXC))), np.zeros((m.N_EXC, m.N_UVA + m.N_INH)) ],
            [ np.zeros((m.N_UVA + m.N_INH, m.N_EXC + m.N_UVA + m.N_INH)) ],
        ]),
    }

    ee_connectivity = np.where(w_r_e[:(m.N_EXC), :(m.N_EXC + m.N_UVA)] > 0, 1, 0)

    pairwise_spk_delays = np.block([
        [int(2.2e-3 / S.DT) * np.ones((m.N_EXC, m.N_EXC)), np.ones((m.N_EXC, m.N_UVA)), int(0.5e-3 / S.DT) * np.ones((m.N_EXC, m.N_INH))],
        [int(0.5e-3 / S.DT) * np.ones((m.N_INH + m.N_UVA, m.N_EXC + m.N_INH + m.N_UVA))],
    ]).astype(int)

    # turn pairwise delays into list of cells one cell is synapsed to with some delay tau
   
    delay_map = {}
    summed_w_r_abs = np.sum(np.stack([np.abs(w_r[syn]) for syn in w_r.keys()]), axis=0)
    for i in range(pairwise_spk_delays.shape[1]):
        cons = summed_w_r_abs[:, i].nonzero()[0]
        delay_map[i] = (pairwise_spk_delays[cons, i], cons)

    def create_prop(prop_exc, prop_inh):
        return cc([prop_exc * np.ones(m.N_EXC + m.N_UVA), prop_inh * np.ones(m.N_INH)])

    c_m = create_prop(m.C_M_E, m.C_M_I)
    g_l = create_prop(m.G_L_E, m.G_L_I)
    e_l = create_prop(m.E_L_E, m.E_L_I)
    v_th = create_prop(m.V_TH_E, m.V_TH_I)
    e_r = create_prop(m.E_R_E, m.E_R_I)
    t_r = create_prop(m.T_R_E, m.T_R_I)

    e_cell_fr_setpoints = np.ones(m.N_EXC) * 5

    sampled_e_cell_rasters = []
    e_cell_sample_idxs = np.sort((np.random.rand(10) * m.N_EXC).astype(int))
    sampled_i_cell_rasters = []
    i_cell_sample_idxs = np.sort((np.random.rand(10) * m.N_INH + m.N_EXC).astype(int))

    w_r_copy = copy(w_r)


    # tracemalloc.start()

    # snapshot = None
    # last_snapshot = tracemalloc.take_snapshot()

    e_cell_pop_fr_measurements = None

    for i_e in range(epochs):

        progress = f'{i_e / epochs * 100}'
        progress = progress[: progress.find('.') + 2]
        print(f'{progress}% finished')

        start = time.time()

        if i_e == m.DROPOUT_ITER:
            w_r_copy['E'][:(m.N_EXC + m.N_UVA + m.N_INH), :m.N_EXC], surviving_cell_indices = dropout_on_mat(w_r_copy['E'][:(m.N_EXC + m.N_UVA + m.N_INH), :m.N_EXC], dropout['E'], min_idx=m.DROPOUT_MIN_IDX, max_idx=m.DROPOUT_MAX_IDX)
            ee_connectivity = np.where(w_r_copy['E'][:(m.N_EXC), :(m.N_EXC + m.N_UVA)] > 0, 1, 0)

        t = np.arange(0, S.T, S.DT)

        ## external currents
        if add_noise:
            i_ext = m.SGM_N/S.DT * np.random.randn(len(t), m.N_EXC + m.N_UVA + m.N_INH) + m.I_EXT_B
        else:
            i_ext = m.I_EXT_B * np.ones((len(t), m.N_EXC + m.N_UVA + m.N_INH))

        ## inp spks
        spks_u_base = np.zeros((len(t), m.N_DRIVING_CELLS + m.N_UVA), dtype=int)

        # trigger inputs
        activation_times = np.zeros((len(t), m.N_DRIVING_CELLS))
        for t_ctr in np.arange(0, S.T, 1./m.DRIVING_HZ):
            activation_times[int(t_ctr/S.DT), :] = 1

        np.concatenate([np.random.poisson(m.DRIVING_HZ * S.DT, size=(len(t), 1)) for i in range(m.N_DRIVING_CELLS)], axis=1)
        spks_u = copy(spks_u_base)
        spks_u[:, :m.N_DRIVING_CELLS] = np.zeros((len(t), m.N_DRIVING_CELLS))
        burst_t = np.arange(0, 5 * int(m.BURST_T / S.DT), int(m.BURST_T / S.DT))

        for t_idx, driving_cell_idx in zip(*activation_times.nonzero()):
            input_noise_t = np.array(np.random.normal(scale=m.INPUT_STD / S.DT), dtype=int)
            try:
                spks_u[burst_t + t_idx + input_noise_t + int(m.INPUT_DELAY / S.DT), driving_cell_idx] = 1
            except IndexError as e:
                pass

        def make_poisson_input(dur=0.2, offset=0.06):
            x = np.zeros(len(t))
            x[int(offset/S.DT):int(offset/S.DT) + int(dur/S.DT)] = np.random.poisson(lam=10 * S.DT, size=int(dur/S.DT))
            return x

        # uva_spks_base = np.random.poisson(lam=20 * S.DT, size=len(t))
        # spks_u[:, m.N_DRIVING_CELLS:(m.N_DRIVING_CELLS + m.N_UVA)] = np.stack([make_poisson_input() for i in range(m.N_UVA)]).T

        ntwk = LIFNtwkG(
            c_m=c_m,
            g_l=g_l,
            e_l=e_l,
            v_th=v_th,
            v_r=e_r,
            t_r=t_r,
            e_s={'E': M.E_E, 'I': M.E_I, 'A': M.E_A},
            t_s={'E': M.T_E, 'I': M.T_E, 'A': M.T_A},
            w_r=w_r_copy,
            w_u=w_u,
            pairwise_spk_delays=pairwise_spk_delays,
            delay_map=delay_map,
        )

        clamp = Generic(v={0: e_l}, spk={})

        # run smln
        rsp = ntwk.run(dt=S.DT, clamp=clamp, i_ext=i_ext, spks_u=spks_u)


        sampled_e_cell_rasters.append(rsp.spks[int((m.INPUT_DELAY + 20e-3)/S.DT):, e_cell_sample_idxs])
        sampled_i_cell_rasters.append(rsp.spks[int((m.INPUT_DELAY + 20e-3)/S.DT):, i_cell_sample_idxs])

        sampled_trial_number = 10
        if i_e % sampled_trial_number == 0 and i_e != 0:
            fig = plt.figure(figsize=(8, 8), tight_layout=True)
            ax = fig.add_subplot()
            base_idx = 0
            for rasters_for_cell_type in [sampled_e_cell_rasters, sampled_i_cell_rasters]:
                for rendition_num in range(len(rasters_for_cell_type)):
                    for cell_idx in range(rasters_for_cell_type[rendition_num].shape[1]):
                        spk_times_for_cell = np.nonzero(rasters_for_cell_type[rendition_num][:, cell_idx])[0]
                        ax.scatter(spk_times_for_cell * S.DT * 1000, (base_idx + cell_idx * len(rasters_for_cell_type) + rendition_num) * np.ones(len(spk_times_for_cell)), s=3, marker='|')
                base_idx += sampled_trial_number * rasters_for_cell_type[0].shape[1]
            ax.set_xlim(0, 150)
            ax.set_xlabel('Time (ms)')
            sampled_e_cell_rasters = []
            sampled_i_cell_rasters = []
            fig.savefig(f'{sampled_cell_output_dir}/sampled_cell_rasters_{int(i_e / sampled_trial_number)}.png')

        scale = 0.8
        gs = gridspec.GridSpec(10, 1)
        fig = plt.figure(figsize=(9 * scale, 25 * scale), tight_layout=True)
        axs = [fig.add_subplot(gs[:2]), fig.add_subplot(gs[2]), fig.add_subplot(gs[3]), fig.add_subplot(gs[4]), fig.add_subplot(gs[5]), fig.add_subplot(gs[6:8]), fig.add_subplot(gs[8:])]

        w_e_e_r_copy = w_r_copy['E'][:m.N_EXC, :m.N_EXC]

        # 0.05 * np.mean(w_e_e_r_copy.sum(axis=1)
        summed_w_bins, summed_w_counts = bin_occurrences(w_e_e_r_copy.sum(axis=1), bin_size=1e-4)
        axs[3].plot(summed_w_bins, summed_w_counts)
        axs[3].set_xlabel('Normalized summed synapatic weight')
        axs[3].set_ylabel('Counts')

        incoming_con_counts = np.count_nonzero(w_e_e_r_copy, axis=1)
        incoming_con_bins, incoming_con_freqs = bin_occurrences(incoming_con_counts, bin_size=1)
        axs[4].plot(incoming_con_bins, incoming_con_freqs)
        axs[4].set_xlabel('Number of incoming synapses per cell')
        axs[4].set_ylabel('Counts')

        graph_weight_matrix(w_r_copy['E'][:m.N_EXC, :(m.N_EXC + m.N_UVA)], 'w_e_e_r\n', ax=axs[5], v_max=m.W_E_E_R_MAX)
        graph_weight_matrix(w_r_copy['I'][:m.N_EXC, (m.N_EXC + m.N_UVA):], 'w_i_e_r\n', ax=axs[6], v_max=m.W_E_I_R)

        spks_for_e_cells = rsp.spks[:, :m.N_EXC]
        spks_for_i_cells = rsp.spks[:, (m.N_EXC + m.N_UVA):(m.N_EXC + m.N_UVA + m.N_INH)]

        spks_received_for_e_cells = rsp.spks_received[:, :m.N_EXC, :m.N_EXC]
        spks_received_for_i_cells = rsp.spks_received[:, (m.N_EXC + m.N_UVA):(m.N_EXC + m.N_UVA + m.N_INH), (m.N_EXC + m.N_UVA):(m.N_EXC + m.N_UVA + m.N_INH)]

        spk_bins, freqs = bin_occurrences(spks_for_e_cells.sum(axis=0), max_val=800, bin_size=1)

        axs[1].bar(spk_bins, freqs, alpha=0.5)
        axs[1].set_xlabel('Spks per neuron')
        axs[1].set_ylabel('Frequency')
        axs[1].set_xlim(-0.5, 30.5)
        # axs[1].set_ylim(0, m.N_EXC + m.N_SILENT)

        raster = np.stack([rsp.spks_t, rsp.spks_c])
        exc_raster = raster[:, raster[1, :] < m.N_EXC]
        inh_raster = raster[:, raster[1, :] >= (m.N_EXC + m.N_UVA)]

        spk_bins_i, freqs_i = bin_occurrences(spks_for_i_cells.sum(axis=0), max_val=800, bin_size=1)

        axs[2].bar(spk_bins_i, freqs_i, color='black', alpha=0.5, zorder=-1)
        axs[2].set_xlim(-0.5, 100)

        axs[0].scatter(exc_raster[0, :] * 1000, exc_raster[1, :], s=1, c='black', zorder=0, alpha=1)
        axs[0].scatter(inh_raster[0, :] * 1000, inh_raster[1, :] - m.N_UVA, s=1, c='red', zorder=0, alpha=1)

        axs[0].set_ylim(-1, m.N_EXC + m.N_INH)
        axs[0].set_xlim(m.INPUT_DELAY * 1000, 500)
        axs[0].set_ylabel('Cell Index')
        axs[0].set_xlabel('Time (ms)')

        for i in range(len(axs)):
            set_font_size(axs[i], 14)
        fig.savefig(f'{output_dir}/{zero_pad(i_e, 4)}.png')

        first_spk_times = process_single_activation(exc_raster, m)

        if i_e > 0:
            # if i_e % 80 == 0 and args.load_run is None:
            #     e_cell_pop_fr_setpoint += m.PROJECTION_NUM * 5

            def burst_filter(spks, filter_size):
                filtered = np.zeros(spks.shape, dtype=bool)
                filter_counter = np.zeros(spks.shape[1:], dtype=int)
                for i_t in range(spks.shape[0]):
                    filtered[i_t, np.bitwise_and(spks[i_t, ...], filter_counter == 0)] = 1
                    filter_counter[filtered[i_t, ...]] = filter_size + 1
                    filter_counter -= 1
                    filter_counter[filter_counter < 0] = 0
                return filtered

            t_steps_in_burst = int(20e-3/S.DT)

            filtered_spks_for_e_cells = burst_filter(spks_for_e_cells, t_steps_in_burst)
            filtered_spks_received_for_e_cells = burst_filter(spks_received_for_e_cells, t_steps_in_burst)

            # # STDP FOR E CELLS: put in pairwise STDP on filtered_spks_for_e_cells
            # stdp_burst_pair_e_e_plus = np.zeros([m.N_EXC , m.N_EXC + m.N_UVA])
            # stdp_burst_pair_e_e_minus = np.zeros([m.N_EXC , m.N_EXC + m.N_UVA])

            # for i_t in range(spks_for_e_cells.shape[0]):
            #     # find E spikes at current time
            #     curr_spks_e = filtered_spks_for_e_cells[i_t, :]
            #     # sparse_curr_spks_e = csc_matrix(curr_spks_e)

            #     ## find E spikes for stdp
            #     stdp_start_ee = i_t - m.CUT_IDX_TAU_PAIR_EE if i_t - m.CUT_IDX_TAU_PAIR_EE > 0 else 0
            #     stdp_end_ee = i_t + m.CUT_IDX_TAU_PAIR_EE if i_t + m.CUT_IDX_TAU_PAIR_EE < spks_for_e_cells.shape[0] else (spks_for_e_cells.shape[0] - 1)

            #     trimmed_kernel_ee_plus = m.KERNEL_PAIR_EE[(M.CUT_IDX_TAU_PAIR_EE + M.TAU_PAIR_EE_CENTER):M.CUT_IDX_TAU_PAIR_EE + (stdp_end_ee - i_t)]
            #     # print(trimmed_kernel_ee_plus)
            #     trimmed_kernel_ee_minus = m.KERNEL_PAIR_EE[M.CUT_IDX_TAU_PAIR_EE - (i_t - stdp_start_ee):(M.CUT_IDX_TAU_PAIR_EE + M.TAU_PAIR_EE_CENTER)]
            #     # print(trimmed_kernel_ee_minus)

            #     for curr_spk_e in curr_spks_e.nonzero()[0]:
            #         sparse_spks_received_e_plus = csc_matrix(filtered_spks_received_for_e_cells[(i_t + M.TAU_PAIR_EE_CENTER):stdp_end_ee, curr_spk_e, :])
            #         sparse_spks_received_e_minus = csc_matrix(filtered_spks_received_for_e_cells[stdp_start_ee:(i_t + M.TAU_PAIR_EE_CENTER), curr_spk_e, :])
            #         stdp_burst_pair_e_e_plus[:, curr_spk_e] += sparse_spks_received_e_plus.T.dot(trimmed_kernel_ee_plus)
            #         stdp_burst_pair_e_e_minus[:, curr_spk_e] += sparse_spks_received_e_minus.T.dot(trimmed_kernel_ee_minus)

            # # E SINGLE-CELL FIRING RATE RULE
            # fr_update_e = 0

            # e_diffs = e_cell_fr_setpoints - np.sum(spks_for_e_cells > 0, axis=0)
            # e_diffs[e_diffs > 0] = 0
            # fr_update_e = e_diffs.reshape(e_diffs.shape[0], 1) * np.ones((m.N_EXC, m.N_EXC + m.N_UVA)).astype(float)


            # # E POPULATION-LEVEL FIRING RATE RULE
            # fr_pop_step = 0

            # if i_e >= m.POP_FR_TRIALS[0] and i_e < m.POP_FR_TRIALS[1]:
            #     if e_cell_pop_fr_measurements is None:
            #         e_cell_pop_fr_measurements = np.sum(spks_for_e_cells)
            #     else:
            #         e_cell_pop_fr_measurements += np.sum(spks_for_e_cells)
            # elif i_e == m.POP_FR_TRIALS[1]:
            #     e_cell_pop_fr_setpoint = e_cell_pop_fr_measurements / (m.POP_FR_TRIALS[1] - m.POP_FR_TRIALS[0])
            # elif i_e > m.POP_FR_TRIALS[1]:
            #     if i_e >= m.DROPOUT_ITER:
            #         current_summed_spks = np.sum(spks_for_e_cells[:, surviving_cell_indices.astype(bool)])
            #     else:
            #         current_summed_spks = np.sum(spks_for_e_cells)
            #     fr_pop_diff = e_cell_pop_fr_setpoint - current_summed_spks
            #     tau_pop = 20
            #     print((-1 + np.exp(fr_pop_diff / tau_pop)) / (1 + np.exp(fr_pop_diff / tau_pop)))
            #     fr_pop_step = m.ETA * m.GAMMA * (-1 + np.exp(fr_pop_diff / tau_pop)) / (1 + np.exp(fr_pop_diff / tau_pop)) * np.ones((m.N_EXC, m.N_EXC + m.N_UVA))
            #     fr_pop_step[:, m.N_EXC:] = 0


            # firing_rate_potentiation = m.ETA * m.ALPHA * fr_update_e
            # stdp_ee_potentiation = m.ETA * m.BETA * stdp_burst_pair_e_e_plus
            # stdp_ee_depression = m.ETA * m.BETA * stdp_burst_pair_e_e_minus

            # w_e_e_hard_bound = m.W_E_E_R_MAX

            # w_r_copy['E'][:m.N_EXC, :(m.N_EXC + m.N_UVA)] += ((stdp_ee_depression + firing_rate_potentiation + fr_pop_step) * w_r_copy['E'][:(m.N_EXC), :(m.N_EXC + m.N_UVA)])
            # w_r_copy['E'][:m.N_EXC, :(m.N_EXC + m.N_UVA)] += (stdp_ee_potentiation * (m.W_E_E_R_MAX * ee_connectivity - w_r_copy['E'][:(m.N_EXC), :(m.N_EXC + m.N_UVA)]))
            
            # w_r_copy['E'][:m.N_EXC, :(m.N_EXC + m.N_UVA)][(w_r_copy['E'][:m.N_EXC, :(m.N_EXC + m.N_UVA)] < m.W_E_E_R_MIN) & ee_connectivity] = m.W_E_E_R_MIN
            # w_r_copy['E'][:m.N_EXC, :(m.N_EXC)][w_r_copy['E'][:m.N_EXC, (m.N_EXC)] > m.W_E_E_R_MAX] = m.W_E_E_R_MAX

            # output weight bound
            # cell_outgoing_weight_totals = w_r_copy['E'][:(m.N_EXC + m.N_SILENT), :(m.N_EXC + m.N_SILENT)].sum(axis=0)
            # rescaling = np.where(cell_outgoing_weight_totals > m.CELL_OUTPUT_MAX, m.CELL_OUTPUT_MAX / cell_outgoing_weight_totals, 1.)
            # w_r_copy['E'][:(m.N_EXC + m.N_SILENT), :(m.N_EXC + m.N_SILENT)] *= rescaling.reshape(1, rescaling.shape[0])

            # print('ei_mean_stdp', np.mean(m.ETA * m.BETA * stdp_burst_pair_e_i))
            # w_r_copy['I'][:(m.N_EXC + m.N_SILENT), (m.N_EXC + m.N_SILENT):] += 1e-4 * m.ETA * m.BETA * stdp_burst_pair_e_i
            # w_r_copy['I'][w_r_copy['I'] < 0] = 0
            # w_r_copy['I'][w_r_copy['I'] > m.W_I_E_R_MAX] = m.W_I_E_R_MAX

        if i_e % 10 == 0:

                mean_initial_first_spk_time = np.nanmean(first_spk_times[400:410])
                mean_final_first_spk_time = np.nanmean(first_spk_times[800:810])
                prop_speed = (80 - 40) / (mean_final_first_spk_time - mean_initial_first_spk_time)

                spiking_idxs = np.nonzero(first_spk_times[400:810])[0]


                spks_for_spiking_idxs = spks_for_e_cells[:, spiking_idxs]

                temporal_widths = []

                for k in range(spks_for_spiking_idxs.shape[1]):
                    temporal_widths.append(np.std(S.DT * np.nonzero(spks_for_spiking_idxs[:, k])[0]))

                avg_temporal_width = np.mean(temporal_widths)
                print(avg_temporal_width)


                base_data_to_save = {
                    'w_e_e': m.W_E_E_R,
                    'w_e_i': m.W_E_I_R,
                    'w_i_e': m.W_I_E_R,
                    'first_spk_times': first_spk_times,
                    'spk_bins': spk_bins,
                    'freqs': freqs,
                    'exc_raster': exc_raster,
                    'inh_raster': inh_raster,
                    'prop_speed': prop_speed,
                    'avg_temporal_width': avg_temporal_width,
                    'stable': len(~np.isnan(first_spk_times[950:1000])) > 30,
                    # 'gs': rsp.gs,
                }


                # if e_cell_fr_setpoints is not None:
                #     base_data_to_save['e_cell_fr_setpoints'] = e_cell_fr_setpoints

                if e_cell_pop_fr_setpoint is not None:
                    base_data_to_save['e_cell_pop_fr_setpoint'] = e_cell_pop_fr_setpoint

                if i_e >= m.DROPOUT_ITER:
                    base_data_to_save['surviving_cell_indices'] = surviving_cell_indices

                # if i_e >= m.DROPOUT_ITER:
                #     update_obj = {
                #         'exc_cells_initially_active': exc_cells_initially_active,
                #         'exc_cells_newly_active': exc_cells_newly_active,
                #         'surviving_cell_indices': surviving_cell_indices,
                #     }
                #     base_data_to_save.update(update_obj)

                # if i_e % 100 == 0:
                #     update_obj = {
                #         'w_r_e': rsp.ntwk.w_r['E'],
                #         'w_r_i': rsp.ntwk.w_r['I'],
                #     }
                #     base_data_to_save.update(update_obj)

                sio.savemat(robustness_output_dir + '/' + f'title_{title}_idx_{zero_pad(i_e, 4)}', base_data_to_save)

        end = time.time()
        secs_per_cycle = f'{end - start}'
        secs_per_cycle = secs_per_cycle[:secs_per_cycle.find('.') + 2]
        print(f'{secs_per_cycle} s')

        plt.close('all')

        # snapshot = tracemalloc.take_snapshot()
        # if last_snapshot is not None:
        #     top_stats = snapshot.compare_to(last_snapshot, 'lineno')
        #     print("[ Top 3 differences ]")
        #     for stat in top_stats[:3]:
        #         print(stat)



def quick_plot(m, run_title='', w_r_e=None, w_r_i=None, n_show_only=None, add_noise=True, dropout={'E': 0, 'I': 0}, e_cell_pop_fr_setpoint=None):
    output_dir_name = f'{run_title}_{time_stamp(s=True)}:{zero_pad(int(np.random.rand() * 9999), 4)}'

    run_test(m, output_dir_name=output_dir_name, n_show_only=n_show_only, add_noise=add_noise, dropout=dropout,
                        w_r_e=w_r_e, w_r_i=w_r_i, epochs=S.EPOCHS, e_cell_pop_fr_setpoint=e_cell_pop_fr_setpoint)

def process_single_activation(exc_raster, m):
    # extract first spikes
    first_spk_times = np.nan * np.ones(m.N_EXC + m.N_UVA)
    for i in range(exc_raster.shape[1]):
        nrn_idx = int(exc_raster[1, i])
        if np.isnan(first_spk_times[nrn_idx]):
            first_spk_times[nrn_idx] = exc_raster[0, i]
    return first_spk_times

def load_previous_run(direc, num):
    file_names = sorted(all_files_from_dir(direc))
    file = file_names[num]
    loaded = sio.loadmat(os.path.join(direc, file))
    return loaded

def clip(f, n=1):
    f_str = str(f)
    f_str = f_str[:(f_str.find('.') + 1 + n)]
    return f_str

w_e_e_r_vals = np.linspace(0.26 * 0.004 * 0.1, 0.26 * 0.004 * 2, 20)
w_e_i_r_vals = np.array([7e-5])
w_i_e_r_vals = np.linspace(1e-5, 7.5e-5, 20)

all_vals = cartesian(w_e_e_r_vals, w_e_i_r_vals, w_i_e_r_vals)

print(all_vals[0])

for idx in range(len(all_vals[0])):
    m = copy(M)

    m.W_E_E_R = all_vals[0][idx]
    m.W_E_I_R = all_vals[1][idx]
    m.W_I_E_R = all_vals[2][idx]

    for j in range(5):
        title = f'{args.title[0]}_idx_{zero_pad(idx, 4)}_repeat_{j}'
        quick_plot(m, run_title=title, dropout={'E': M.DROPOUT_SEV, 'I': 0})


