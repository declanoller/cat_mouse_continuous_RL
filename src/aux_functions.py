import RunTools as rt
import os, json
import path_utils, plot_tools
import torch
from CatMouse_DDPG import CatMouse_DDPG

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
		#cm.plot_VF(save_plot=True, show_plot=False, cat_pos=[1,0])
		#cm.plot_VF(save_plot=True, show_plot=False, cat_pos=[0,1])
		#cm.plot_VF(save_plot=True, show_plot=False, cat_pos=[-1,0])
		#cm.plot_VF(save_plot=True, show_plot=False, cat_pos=[0,-1])
		pass
	except:
		print('Problem plotting VFs')

	return(rt.smooth_data(cm.R_train_hist))


@path_utils.timer
def ramp_difficulty(train_class, **kwargs):

	'''
	This should hopefully "ramp up" the difficulty, letting it
	solve it for the easy case, and then adapt to the harder cases.
	'''

	base_dir = os.path.join(path_utils.get_output_dir(), 'ramp_difficulty_{}'.format(path_utils.get_date_str()))
	os.mkdir(base_dir)

	kwargs['base_dir'] = base_dir

	cm = train_class(**kwargs)

	#cat_speed_rels = [3.0, 3.1, 3.2, 3.3, 3.35, 3.4, 3.45, 3.5, 3.55, 3.6, 3.7, 3.8, 3.9, 4.0]
	N_eps_chunk = kwargs.get('N_eps', 5000)
	cat_speed_rels = {
						3.1 : 1*N_eps_chunk,
						3.2 : 1*N_eps_chunk,
						3.3 : 1*N_eps_chunk,
						3.4 : 2*N_eps_chunk,
						3.5 : 2*N_eps_chunk,
						3.6 : 2*N_eps_chunk,
						3.65 : 2*N_eps_chunk,
						3.7 : 2*N_eps_chunk,
						3.75 : 2*N_eps_chunk,
						3.8 : 2*N_eps_chunk,
					}


	c_s_r_fname = os.path.join(cm.dir, 'c_s_r_list.json')
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

	cm.save_model()



def load_model_run_episode(run_dir, **kwargs):


	model_dir = os.path.join(run_dir, 'saved_models')
	assert os.path.exists(model_dir), 'model dir must exist!'

	with open(os.path.join(run_dir, 'params_passed.json'), 'r') as f:
		cm_kwargs = json.load(f)

	cm_kwargs['dir'] = run_dir
	cm = CatMouse_DDPG(**cm_kwargs)
	cm.agent.set_mouse_speed_rad_div(5.0)

	csr_fname = os.path.join(run_dir, 'c_s_r_list.json')
	if os.path.exists(csr_fname):
		with open(csr_fname, 'r') as f:
			csr_dict = json.load(f)

		top_csr = max([float(x) for x in csr_dict['cat_speed_rels'].keys()])
		print(f'Top csr is {top_csr}')
		cm.agent.set_cat_speed(3.7)

	print('\nLoading model...')
	cm.load_model()

	for i in range(100):
		print('Doing no-train episode...')
		ret_dict = cm.no_train_episode(200)
		tot_steps = ret_dict['tot_steps']
		if ret_dict['mouse_caught'] == 'escaped':
			print(f'{tot_steps} total steps to escape')
		if ret_dict['mouse_caught'] == 'caught':
			print(f'{tot_steps} total steps to catch')
			print('Making traj gif...')
			gif_len = 0.01*ret_dict['tot_steps']

			plot_tools.make_traj_gif(ret_dict['traj_fname'], length=gif_len)
			break















#
