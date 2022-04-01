import numpy as np
import datetime

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


def time_stamp(s=False):
    if s:
        return datetime.datetime.now().strftime('%Y-%m-%d--%H:%M--%S')
    else:
        return datetime.datetime.now().strftime('%Y-%m-%d--%H:%M')


def zero_pad(arg, size):
    s = str(arg)
    while len(s) < size:
        s = '0' + s
    return s


def outer_product_n_dim(*args):
    params = [p for p in args]
    outer_product = np.meshgrid(*params, sparse=False, indexing='ij')
    return np.stack([p_vals.flatten() for p_vals in outer_product], axis=1)


# returns a subset of dataframe
def select(df, selection):
    criteria = []
    for col in selection:
        criteria.append(df[col] == selection[col])
    return df[np.all(criteria, axis=0)]


# returns a list of unique values in the given Pandas dataframe for each column name specified 
def to_unique_vals(df, col_names):
    if type(col_names) is str:
        col_names = [col_names]
    return tuple([df[col_name].unique() for col_name in col_names])
