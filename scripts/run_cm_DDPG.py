
import path_utils

from CatMouse_DDPG import CatMouse_DDPG
import aux_functions
import RunTools as rt




########################################### Ramp diff
center_dict = {
	'N_eps' : 5000,
	'noise_sigma' : 0.5,
	'max_ep_steps' : 100,
	'N_steps' : 100,
	'decay_noise' : True,
	'reset_buffer' : True
}

aux_functions.ramp_difficulty(CatMouse_DDPG, **center_dict)

exit()




#################################### DDPG difficulty tests

center_dict = {
	'train_class' : CatMouse_DDPG,
	'N_eps' : 3000,
	'noise_sigma' : 0.5,
	'max_ep_steps' : 100,
	'N_steps' : 100,
	'decay_noise' : True,
	'cat_speed_rel' : 3.2
}


vary_dict = {

}

rt.hyperparam_search_const(
								aux_functions.run_train_sequence,
								center_dict,
								vary_dict,
								path_utils.get_output_dir(),
								N_runs=8,
								save_R = True,
								center_run = True
							)

exit()



















#
