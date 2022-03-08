import os


def filter_by_name_frags(name, name_frags, in_order=True):
	'''
	Yields 'name' if it separately contains all the strings within 'name_frags'. If 'in_order' is True, name fragements must be in order.
	Parameters
	----------
	name : string
		String to be returned if it contains all of 'name_frags'
	name_frags : list of strings
		Name fragments that 'name' must contain to be yielded
	in_order : boolean
		If true, 'name' must contain 'name_frags' in the order they are specified
	Returns
	-------
	Yields 'name' or nothing
	'''

	# simply yield name if there are no name fragments with which to filter
	if len(name_frags) == 0:
		yield name
	# working name is pruned as fragments are found within name to ensure fragments are found in order
	working_name = name

	# split any name fragments with '*' into two separate name fragments
	expanded_name_frags = []
	for frag in name_frags:
		expanded_name_frags += frag.split('*')

	for i, frag in enumerate(expanded_name_frags):
		try:
			idx = working_name.index(frag)
			frag_end_idx = idx + len(frag)
			if in_order:
				# if fragment is found in working name, prune working_name if order matters
				working_name = working_name[frag_end_idx:]
			if i == len(expanded_name_frags) - 1:
				# last fragment has been found; yield name
				yield name
		except ValueError:
			# name fragment does not exist within working_name; name is filtered out
			break


def filter_list_by_name_frags(l, name_frags, in_order=True):
	'''
	Filters strings contained in list 'l' by 'name_frags'. Each string within 'l' must contain
	the elements of 'name_frags' in order if 'in_order' is True or out of order if False
	Parameters
	----------
	l : list of strings
		List of strings to be filtered
	name_frags : list of strings
		Name fragments that each element of 'l' must contain to be yielded
	in_order : boolean
		If true, elements of 'l' must contain 'name_frags' in the order 'name_frags' specifies
	Returns
	-------
	Function itself is a generator that yields elements of 'l' that pass filtration
	'''
	for name in l:
		for matching_name in filter_by_name_frags(name, name_frags, in_order=in_order):
			yield matching_name


def all_in_dir(path_to_dir):
	'''
	Returns the names of all files and directories within the directory specified by the path to 'path_to_dir'
	Parameters
	----------
	path_to_dir : string
		Relative path to directory to read
	Returns
	-------
	List of strings of all files and directories in specified directory
	'''
	return os.listdir(path_to_dir)


def all_files_from_dir(path_to_dir):
	'''
	Similar to 'all_in_dir' but only returns files.
	Parameters
	----------
	path_to_dir : string
		Relative path to directory to read
	'''
	all_entries = all_in_dir(path_to_dir)
	return [name for name in all_entries if not os.path.isdir(os.path.join(path_to_dir, name))]


def all_files_with_name_frags(path_to_dir, name_frags, in_order=True):
	'''
	Composes 'all_files_from_dir' and 'filter_list_by_name_frags' to read all files from a directory and filter them
	Parameters
	----------
	path_to_dir : string
		Relative path from 'curr_file' to directory to read
	name_frags : list of strings
		Name fragments that each file in specified directory must contain to be yielded
	in_order : boolean
		If true, files in specified directory must contain 'name_frags' in the order 'name_frags' specifies
	Returns
	-------
	File names in specified directory that pass filtration
	'''
	if not isinstance(name_frags, list):
		name_frags = [name_frags]
	all_files = all_files_from_dir(path_to_dir)
	return [f for f in filter_list_by_name_frags(all_files, name_frags, in_order=in_order)]
