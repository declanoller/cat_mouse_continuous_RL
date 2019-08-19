import RunTools as rt


def run_train_sequence(**kwargs):

	'''
	For running a training sequence with a new agent, with new parameters. Used
	with rt.hyperparam_search_const.

	You have to pass a train_class (ie., CatMouse_DDPG.CatMouse_DDPG)
	'''

	train_class = kwargs.get('train_class', None)
	assert train_class is not None, 'Must provide a train_class!'

	cm = train_class(**kwargs)

	cm.train(kwargs.get('N_eps', 20), kwargs.get('N_steps', 500))
	try:
		cm.plot_VF(save_plot=True, show_plot=False, cat_pos=[1,0])
		cm.plot_VF(save_plot=True, show_plot=False, cat_pos=[0,1])
		cm.plot_VF(save_plot=True, show_plot=False, cat_pos=[-1,0])
		cm.plot_VF(save_plot=True, show_plot=False, cat_pos=[0,-1])
	except:
		print('Problem plotting VFs')

	return(rt.smooth_data(cm.R_train_hist))



def ramp_difficulty(train_class, **kwargs):

	'''
	This should hopefully "ramp up" the difficulty, letting it
	solve it for the easy case, and then adapt to the harder cases.
	'''

	base_dir = os.path.join(os.path.join(os.path.dirname(__file__), 'misc_runs'), 'ramp_difficulty_{}'.format(fst.getDateString()))
	os.mkdir(base_dir)

	kwargs['base_dir'] = base_dir

	cm = train_class(**kwargs)

	#cat_speed_rels = [3.0, 3.1, 3.2, 3.3, 3.35, 3.4, 3.45, 3.5, 3.55, 3.6, 3.7, 3.8, 3.9, 4.0]
	N_eps_chunk = 5000
	cat_speed_rels = {
						3.0 : 1*N_eps_chunk,
						3.2 : 2*N_eps_chunk,
						3.3 : 2*N_eps_chunk,
						3.4 : 2*N_eps_chunk,
						3.5 : 4*N_eps_chunk
					}


	c_s_r_fname = os.path.join(cm.dir, 'c_s_r_list.txt')
	with open(c_s_r_fname, 'w+') as f:
		json.dump({'cat_speed_rels' : cat_speed_rels}, f, indent=4)

	for i, (c_s, N_eps) in enumerate(cat_speed_rels.items()):

		print(f'\nRunning with cat_speed_rel = {c_s} now!\n')

		if kwargs.get('reset_buffer', False):
			cm.ER.reset_buffer()

		cm.agent.set_cat_speed(c_s)
		if i==0:
			cm.train(N_eps, kwargs.get('N_steps', 500), reset_hist=True)
		else:
			cm.train(N_eps, kwargs.get('N_steps', 500), reset_hist=False)

	cm.train_epochs.append(cm.total_step)
	cm.plot_train(save_plot=True, show_plot=False)
	return(rt.smooth_data(cm.R_train_hist))




#
