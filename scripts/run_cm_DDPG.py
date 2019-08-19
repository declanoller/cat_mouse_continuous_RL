
import path_utils

from CatMouse_DDPG import CatMouse_DDPG
import aux_functions
import RunTools as rt
'''
'N_eps' : 5000,
'noise_sigma' : 0.5,
'''

'''center_dict = {
	'train_class' : CatMouse_DDPG,
	'N_eps' : 3000,
	'noise_sigma' : 0.5,
	'max_ep_steps' : 100,
	'N_steps' : 100,
	'decay_noise' : True,
	'hidden_size' : 300,
	'max_buffer_size' : 10**4,
	'reset_buffer' : True,
	'tau' : 0.01,
	'cat_speed_rel' : 3.0
}'''

center_dict = {
	'train_class' : CatMouse_DDPG,
	'N_eps' : 5000,
	'noise_sigma' : 0.5,
	'max_ep_steps' : 100,
	'N_steps' : 100,
	'decay_noise' : True,
	'cat_speed_rel' : 3.0
}


vary_dict = {
	'tau' : [0.1, 0.01],
	'noise_sigma' : [2.0, 0.1, 0.0],
	'ER_batch_size' : [16, 256],
	'optim' : ['rmsprop']

}

rt.hyperparam_search_const(
								aux_functions.run_train_sequence,
								center_dict,
								vary_dict,
								path_utils.get_output_dir(),
								N_runs=6,
								save_R = True,
								center_run = True
							)


exit()





###########################################
center_dict = {
	'noise_sigma' : 0.5,
	'max_ep_steps' : 100,
	'N_steps' : 100,
	'decay_noise' : True,
	'hidden_size' : 300,
	'max_buffer_size' : 10**4,
	'reset_buffer' : True,
	'tau' : 0.01
}

ramp_difficulty(**center_dict)

exit()











#############################################
center_dict = {
	'N_eps' : 25000,
	'noise_sigma' : 0.5,
	'max_ep_steps' : 100,
	'N_steps' : 100,
	'decay_noise' : True,
	'hidden_size' : 300,
	'max_buffer_size' : 10**4,
	'reset_buffer' : True,
	'tau' : 0.01,
	'cat_speed_rel' : 3.5
}

vary_dict = {
}

HP_search(center_dict, vary_dict, N_runs=3, center_run=True)


#############################################
center_dict = {
	'N_eps' : 25000,
	'noise_sigma' : 0.5,
	'max_ep_steps' : 100,
	'N_steps' : 100,
	'decay_noise' : True,
	'hidden_size' : 300,
	'max_buffer_size' : 10**4,
	'reset_buffer' : True,
	'tau' : 0.001,
	'cat_speed_rel' : 3.5
}

vary_dict = {
}

HP_search(center_dict, vary_dict, N_runs=3, center_run=True)


#############################################
center_dict = {
	'N_eps' : 25000,
	'noise_sigma' : 0.5,
	'max_ep_steps' : 100,
	'N_steps' : 100,
	'decay_noise' : True,
	'hidden_size' : 300,
	'max_buffer_size' : 10**4,
	'reset_buffer' : True,
	'tau' : 0.01,
	'cat_speed_rel' : 3.7
}

vary_dict = {
}

HP_search(center_dict, vary_dict, N_runs=3, center_run=True)




exit()











#
