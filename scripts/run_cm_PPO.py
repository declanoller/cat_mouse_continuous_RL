
import path_utils

from CatMouse_PPO import CatMouse_PPO
import aux_functions
import RunTools as rt

center_dict = {
	'train_class' : CatMouse_PPO,
	'N_eps' : 3000,
	'N_steps' : 100,
	'max_ep_steps' : 100,
	'optim' : 'Adam',
	'cat_speed_rel' : 3.0,
	'noise_sigma' : 0.8,
	'explore_steps' : 0,
	'sigma_min' : 10**-3,
	'beta_entropy' : 10**-4

}

vary_dict = {
}

rt.hyperparam_search_const(
								aux_functions.run_train_sequence,
								center_dict,
								vary_dict,
								path_utils.get_output_dir(),
								N_runs=3,
								save_R = True,
								center_run = True
							)


exit()




###########################################
center_dict = {
	'N_eps' : 5000,
	'max_ep_steps' : 100,
	'N_steps' : 100,
	'hidden_size' : 200,
	'sigma_min' : 10**-2,
	'beta_entropy' : 10**-3,
	'N_batch' : 30
}

ramp_difficulty(**center_dict)
exit()



#############################################
center_dict = {
	'optim' : 'Adam',
	'cat_speed_rel' : 3.0,
	'noise_sigma' : 0.5,
	'explore_steps' : 0,
	'max_ep_steps' : 100,
	'N_eps' : 3000,
}

vary_dict = {
	'decay_noise' : [True]
}

HP_search(center_dict, vary_dict, N_runs=3, center_run=True)
exit()

















#
