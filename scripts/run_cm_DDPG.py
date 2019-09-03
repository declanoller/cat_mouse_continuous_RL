import path_utils

from CatMouse_DDPG import CatMouse_DDPG
import aux_functions


run_kwargs = {
	'noise_sigma' : 0.5,
	'max_ep_steps' : 100,
	'N_steps' : 100,
	'cat_speed_rel' : 3.0,
	'decay_noise' : True
}

cm = CatMouse_DDPG(**run_kwargs)
cm.train(1000, 100)

exit()


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





#################################### DDPG test noise
center_dict = {
	'train_class' : CatMouse_DDPG,
	'N_eps' : 3000,
	'max_ep_steps' : 100,
	'N_steps' : 100,
	'cat_speed_rel' : 3.2,
	'noise_sigma' : 0.5,
	'noise_theta' : 0.15
}

vary_dict = {
	'noise_sigma' : [0.05, 0.25, 0.8, 1.0],
	'noise_theta' : [0.01, 0.8],
}

aux_functions.hyperparam_search_const(
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
