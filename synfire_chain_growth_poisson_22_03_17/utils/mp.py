from .general import map_to_list
import multiprocessing as mp

def func_wrapper(args):
	func = args[0]
	args = args[1:]
	return func(*args)


def map_parallel(func, args_list, cores=None):
	cores = mp.cpu_count() if cores is None else cores
	args_list_with_func = map_to_list(lambda args: [func] + args, args_list)
	results = []

	for completed in range(0, len(args_list), cores):
		pool = mp.Pool(cores)
		partial_results = pool.map(func_wrapper, args_list_with_func[completed:(completed + cores)])
		pool.close()
		pool.join()
		results.append(partial_results)
	results = [res for partial_results in results for res in partial_results]
	return results

