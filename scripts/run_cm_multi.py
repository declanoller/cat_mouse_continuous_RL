
import path_utils

from CatMouse_DDPG import CatMouse_DDPG
from CatMouse_A2C import CatMouse_A2C
import aux_functions
import RunTools as rt





######################################## A2C difficulty tests


center_dict = {
	'train_class' : CatMouse_A2C,
	'N_eps' : 20000,
	'N_steps' : 100,
	'max_ep_steps' : 100,
	'gamma' : 0.99,
	'LR' : 2*10**-4,
	'N_batch' : 5,
	'optim' : 'Adam',
	'beta_entropy' : 10**-5,
	'sigma_min' : 10**-5,
	'clamp_grad' : 10,
	'cat_speed_rel' : 3.2,
	'noise_sigma' : 0.5
}

vary_dict = {
	'cat_speed_rel' : [3.3, 3.4],
	'LR' : [2*10**-3]
}


rt.hyperparam_search_const(
								aux_functions.run_train_sequence,
								center_dict,
								vary_dict,
								path_utils.get_output_dir(),
								N_runs=5,
								save_R = True,
								center_run = True
							)




#################################### DDPG difficulty tests

center_dict = {
	'train_class' : CatMouse_DDPG,
	'N_eps' : 20000,
	'noise_sigma' : 0.5,
	'max_ep_steps' : 100,
	'N_steps' : 100,
	'decay_noise' : True,
	'cat_speed_rel' : 3.0
}


vary_dict = {
	'cat_speed_rel' : [3.0, 3.1, 3.2, 3.3, 3.4, 3.5]
}

rt.hyperparam_search_const(
								aux_functions.run_train_sequence,
								center_dict,
								vary_dict,
								path_utils.get_output_dir(),
								N_runs=5,
								save_R = True,
								center_run = True
							)

exit()










#
