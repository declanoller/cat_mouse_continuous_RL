import path_utils, plot_tools
import os, json
from copy import deepcopy
import pprint as pp
from CatMouse_DDPG import CatMouse_DDPG

def run_train_sequence(**kwargs):

	'''
	For running a training sequence with a new agent, with new parameters. Used
	with hyperparam_search_const().

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

	return(plot_tools.smooth_data(cm.R_train_hist))


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




def hyperparam_search_const(run_fn, center_dict, vary_param_dict, output_dir, **kwargs):

	'''
	This is when you have a "center" dict of kwargs (that could be seen
	as the "control"), and you want to vary other parameters one at a time,
	while keeping the others constant.

	So pass it something like: center_dict = {'param1' : 5, 'param2' : 'g', 'param3' : 492},
	and vary_param_dict = {'param1' : [2, 10], 'param2' : ['h', 'q', 'z']}, and it will
	try the center value, and then 5 other variants.

	The function that will be run, run_fn, has to take only kwargs. You'll probably
	have to write an external wrapper for it, if you want to do something like
	create an object of a class and run a few functions for it.

	output_dir is the path you want this function to build the whole tree structure
	in (it will create run_dir for this).

	This isn't fully general at the moment because of the save_param_R_curves.
	It should be expanded so that fn is passed as an arg.
	'''

	# How many runs of each set of params it will do
	N_runs = kwargs.get('N_runs', 1)
	save_R = kwargs.get('save_R', False)

	# Create the run_dir for all the vary_params
	run_dir = os.path.join(output_dir, 'vary_param_{}'.format(path_utils.get_date_str()))
	os.mkdir(run_dir)

	# Whether to do the center run or not
	center_run = kwargs.get('center_run', True)
	if center_run:
		print('\n\nRunning with center...')
		pp.pprint(center_dict, width=1)

		# Create dir for center run
		center_dir = os.path.join(run_dir, 'center_runs')
		os.mkdir(center_dir)

		param_R_curves = []

		# Deepcopy the center_dict, so varying the copy doesn't mess up anything.
		d = deepcopy(center_dict)
		d['base_dir'] = center_dir
		for i in range(N_runs):
			R_curve = run_fn(**d)
			param_R_curves.append(R_curve)

		if save_R:
			save_param_R_curves(param_R_curves, d, center_dir)


	for param, param_list in vary_param_dict.items():
		print('\n\nNow varying parameter: ', param)
		vary_dir = os.path.join(run_dir, 'vary_{}'.format(param))
		os.mkdir(vary_dir)

		for v in param_list:

			param_R_curves = []

			d = deepcopy(center_dict)
			d[param] = v
			d['base_dir'] = vary_dir
			d['fname_note'] = '{}={}'.format(param, v)

			print('\n\nNow running with parameter {} = {}\n'.format(param, v))
			pp.pprint(d, width=1)

			for i in range(N_runs):
				R_curve = run_fn(**d)
				param_R_curves.append(R_curve)

			if save_R:
				save_param_R_curves(param_R_curves, {param : v}, vary_dir)




def save_param_R_curves(param_R_curves, param_val_dict, out_dir):

	if 'base_dir' in param_val_dict.keys():
		param_val_dict = deepcopy(param_val_dict)
		del param_val_dict['base_dir'] # To get rid of this for the center dict

	plt.close('all')
	fig = plt.figure(figsize=(10,6))

	plt.xlabel('Episode')

	plt.ylabel('Total episode R')

	for r_c in param_R_curves:
		plt.plot(*r_c)



	plt.suptitle(path_utils.linebreak_every_n_spaces(path_utils.param_dict_to_label_str(param_val_dict)))
	plt.tight_layout()
	fig.subplots_adjust(top=0.85)

	plt.savefig(os.path.join(out_dir, '{}.png'.format(path_utils.param_dict_to_fname_str(param_val_dict))))















#
