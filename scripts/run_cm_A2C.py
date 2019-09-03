
import path_utils

from CatMouse_A2C import CatMouse_A2C
import aux_functions


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
	'cat_speed_rel' : [3.3, 3.4]
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










vary_dict = {
	'init_weight_bd' : [10**-2, 10**-3],
	'gamma' : [0.999, 0.9999]
}



cm = CatMouse_batch(
			init_weights=True,
			gamma=0.999,
			LR=2*10**-4,
			beta_entropy=10**-4,
			sigma_min=10**-4,
			optim='Adam',
			N_batch = 5,
			clamp_grad=10.0,
			actor_tanh=True,
			cat_speed_rel=3.8,
			init_weight_bd=3*10**-3,
			)


cm.train(5, 500)
cm.plot_train(save_plot=True, show_plot=False)
cm.plot_VF(save_plot=True, show_plot=False, cat_pos=[1,0])
cm.plot_VF(save_plot=True, show_plot=False, cat_pos=[0,1])
cm.plot_VF(save_plot=True, show_plot=False, cat_pos=[-1,0])
cm.plot_VF(save_plot=True, show_plot=False, cat_pos=[0,-1])

exit()



















#
