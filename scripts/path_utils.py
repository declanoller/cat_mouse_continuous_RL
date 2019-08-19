import os, sys, functools, time
from datetime import datetime
from copy import deepcopy

'''
For adding the dirs to the system path, so we can reference
classes from wherever we run a script. Import this first for any
file you want to use it for.

src/ classes will only be called from scripts, so assume that this has
already been called and dirs added to the system path.
'''


ROOT_DIR = os.path.join(os.path.dirname(__file__), '../')
SRC_DIR = os.path.join(ROOT_DIR, 'src')
MISC_DIR = os.path.join(ROOT_DIR, 'misc')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output/misc_runs')

sys.path.append(ROOT_DIR)
sys.path.append(SRC_DIR)

def get_output_dir():
    return OUTPUT_DIR

def get_src_dir():
    return SRC_DIR

def get_misc_dir():
    return MISC_DIR

def timer(func):
    # Print the runtime of the decorated function
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        print(f'\n\nFinished function {func.__name__!r} in {run_time:.2f} secs\n')
        return value
    return wrapper_timer



################### Useful formatting functions

def get_date_str():
    # Returns the date and time for labeling output.
    # -4 to only take two second decimal places.
	return datetime.now().strftime('%d-%m-%Y_%H-%M-%S.%f')[:-4]


def dict_to_str_list(dict):
	pd_copy = deepcopy(dict)
	for k,v in pd_copy.items():
		if isinstance(v, float):
			if abs(v) > 10**-4:
				pd_copy[k] = '{:.3f}'.format(v)
			else:
				pd_copy[k] = '{:.2E}'.format(v)

	params = [f'{k}={v}' for k,v in pd_copy.items() if v is not None]
	return params


def param_dict_to_fname_str(param_dict):
	# Creates a string that can be used as an fname, separated by
	# underscores. If a param has the value None, it isn't included.
	params = dict_to_str_list(param_dict)
	return '_'.join(params)


def param_dict_to_label_str(param_dict):
	# Creates a string that can be used as an fname, separated by
	# commas. If a param has the value None, it isn't included.
	params = dict_to_str_list(param_dict)
	return ', '.join(params)


def linebreak_every_n_spaces(s, n=3):

	# This is to turn every nth space into a newline,
	# for stuff like plotting.
	counter = 0
	while True:
		if counter > 1:
			t = s.replace(' ', '\n', 1)
			counter = 0
		else:
			t = s.replace(' ', 'PLACEHOLDERXYZ', 1)
			counter += 1

		if t == s:
			break
		else:
			s = t

	s = s.replace('PLACEHOLDERXYZ', ' ')
	return(s)

#
