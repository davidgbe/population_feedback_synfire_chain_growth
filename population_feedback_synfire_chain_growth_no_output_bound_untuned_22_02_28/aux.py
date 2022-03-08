# miscellaneous useful functions and classes
import numpy as np
import os
from copy import deepcopy as copy


class Generic(object):
    """Class for generic object."""
    
    def __init__(self, **kwargs):
        
        for k, v in kwargs.items():
            self.__dict__[k] = v


def c_tile(x, n):
    """Create tiled matrix where each of n cols is x."""
    return np.tile(x.flatten()[:, None], (1, n))


def r_tile(x, n):
    """Create tiled matrix where each of n rows is x."""
    return np.tile(x.flatten()[None, :], (n, 1))


def burst_count(ndarr):
	cnts_per_nrn = ndarr.sum(axis=0)
	return cnts_per_nrn, cnts_per_nrn.mean(), cnts_per_nrn.std()


def uncertainty_plot(ax, x, y, y_stds):
	ax.plot(x, y)
	ax.fill_between(x, y - y_stds, y + y_stds)


def bin_occurrences(occurrences, min_val=0, max_val=None, bin_size=1):
    scaled_occurrences = ((occurrences - min_val) / bin_size).astype(int)

    if max_val is None:
        max_val = occurrences.max()

    max_idx = int(np.ceil((max_val - min_val) / bin_size)) + 1

    binned = np.zeros(max_idx, dtype=int)
    for i, n in enumerate(scaled_occurrences):
        if n >= max_idx or n < 0:
            continue
            # raise IndexError(f'val {occurrences[i]} is out of bounds for min {min_val} and max {max_val}')
        binned[n] += 1
    return np.arange(max_idx) * bin_size + min_val, binned


def rand_n_ones_in_vec_len_l(n, l):
    if n > l:
        raise ValueError('n cannot be greater than l')
    vec = np.concatenate([np.ones(n, int), np.zeros(l - n, int)])
    return vec[np.random.permutation(l)]


def rand_per_row_mat(n, shape):
    return np.stack([rand_n_ones_in_vec_len_l(n, shape[1]) for i in range(shape[0])])


def mat_1_if_under_val(val, shape):
    return np.where(np.random.rand(*shape) < val, 1, 0)


def gaussian(shape, mean, std):
    return np.random.normal(loc=mean, scale=std, size=shape)


def gaussian_if_under_val(val, shape, mean, std):
    return np.where(np.random.rand(*shape) < val, np.random.normal(loc=mean, scale=std, size=shape), 0)


def dropout_on_mat(mat, percent, min_idx=0, max_idx=None):
    if max_idx is None:
        max_idx = mat.shape[1]

    num_idxs_in_bounds = max_idx - min_idx

    survival_indices = rand_n_ones_in_vec_len_l(int((1. - percent) * num_idxs_in_bounds), num_idxs_in_bounds)
    survival_indices = np.concatenate([np.ones(min_idx), survival_indices, np.ones(mat.shape[1] - max_idx)])

    m = copy(mat)
    m[:, survival_indices == 0] = 0
    return m, survival_indices


def safe_apply_stat(data, metric):
    out = []
    l = np.max([len(d) for d in data])
    for i in range(l):
        all_data_i = []
        for j, arr in enumerate(data):
            if i < len(arr):
                all_data_i.append(arr[i])
        out.append(metric(all_data_i))
    return np.array(out)


def flatten(a):
    return [x for y in a for x in y]


def map_to_list(func, l):
    '''
    Maps the list 'l' through the function 'func'
    Parameters
    ----------
    func : function
        Takes a single argument of type of 'l'
    l : list
    '''
    return list(map(func, l))


def reduce_mult(l):
    return functools.reduce(lambda e1, e2: e1 * e2, l, 1)


# multidimensional generalization of a cartesian proces
# given [2, 4, 6] and [2, 5, 8, 9] generates
# [[2, 2, 2, 2, 4, 4, 4, 4, 6, 6, 6, 6], [2, 5, 8, 9, 2, 5, 8, 9, 2, 5, 8, 9]]
def cartesian(*arrs):
    domain = map_to_list(lambda a: len(a), arrs)
    coordinate_lists = []
    for i, dim in enumerate(domain):
        coords = []
        mult = 1
        if i != len(domain) - 1:
            mult = reduce_mult(domain[i+1:])
        for e in arrs[i]:
            coords += (mult * [e])
        repeat_factor = reduce_mult(domain[0:i])
        if repeat_factor > 0:
            coords *= repeat_factor
        coordinate_lists.append(coords)
    return coordinate_lists

