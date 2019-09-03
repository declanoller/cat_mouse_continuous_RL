# Import first
import path_utils

import numpy as np
import matplotlib.pyplot as plt

# torch
import torch
from torch import nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
from torch.nn.functional import softplus, relu6

# generic packages
import os, json, random, shutil
import traceback as tb
from copy import deepcopy

# Ones specific to this project
from CatMouseAgent import CatMouseAgent
from OrnsteinUhlenbeckNoise import OrnsteinUhlenbeckNoise
import plot_tools


'''

DDPG implementation.


'''


default_kwargs = {
	'gamma' : 0.99,
	'optim' : 'Adam',
	'LR_actor' : 1*10**-4,
	'LR_critic' : 1*10**-3,
	'clamp_grad' : 10000.0,
	'hidden_size' : 200,
	'init_weights' : True,
	'base_dir' : path_utils.get_output_dir(),
	'noise_sigma' : 0.2,
	'noise_theta' : 0.15,
	'noise_dt' : 10**-2,
	'noise_mu' : 'zero',
	'noise_decay_limit' : 10**-3,
	'max_buffer_size' : 10**6,
	'ER_batch_size' : 64,
	'tau' : 0.001,
	'decay_noise' : True,
	'noise_method' : 'OU'
}

class CatMouse_DDPG:


	def __init__(self, **kwargs):
		global default_kwargs
		#self.agent = CatMouseAgent()
		self.agent = CatMouseAgent(**kwargs)

		self.N_state_terms = len(self.agent.getStateVec())
		self.N_actions = self.agent.N_actions

		for k,v in kwargs.items():
			if k in default_kwargs.keys():
				default_kwargs[k] = v
			else:
				#assert k in default_kwargs.keys(), 'key error'
				print(f'Passed parameter {k} not in default_kwargs dict! Check')

		self.fname_base = 'CatMouse_DDPG_' + path_utils.get_date_str()

		if kwargs.get('dir', None) is None:
			self.dir = os.path.join(default_kwargs['base_dir'], self.fname_base)
			os.mkdir(self.dir)
			print(f'\n\nMade dir {self.dir} for run...\n\n')
		else:
			self.dir = kwargs.get('dir', None)

		self.save_params_json(kwargs, 'params_passed')
		self.save_params_json(default_kwargs, 'params_all')

		self.gamma = default_kwargs['gamma']
		self.optim = default_kwargs['optim']
		self.LR_actor = default_kwargs['LR_actor']
		self.LR_critic = default_kwargs['LR_critic']
		self.hidden_size = default_kwargs['hidden_size']
		self.init_weights = default_kwargs['init_weights']
		self.clamp_grad = default_kwargs['clamp_grad']
		self.noise_sigma = default_kwargs['noise_sigma']
		self.noise_theta = default_kwargs['noise_theta']
		self.noise_dt = default_kwargs['noise_dt']
		self.noise_mu = default_kwargs['noise_mu']
		self.noise_decay_limit = default_kwargs['noise_decay_limit']
		self.max_buffer_size = default_kwargs['max_buffer_size']
		self.ER_batch_size = default_kwargs['ER_batch_size']
		self.tau = default_kwargs['tau']
		self.decay_noise = default_kwargs['decay_noise']
		self.noise_method = default_kwargs['noise_method']


		self.setup_NN()
		#self.tb_writer = SummaryWriter()
		self.dat_fname_file = None

		if 'base_dir' in default_kwargs.keys():
			no_base_dir_kwargs = deepcopy(default_kwargs)
			del no_base_dir_kwargs['base_dir'] # To get rid of this for the center dict
		self.train_plot_title = path_utils.linebreak_every_n_spaces(path_utils.param_dict_to_label_str(no_base_dir_kwargs))


		self.ER = ExpReplay(self.max_buffer_size, self.ER_batch_size)


	def save_params_json(self, d, fname):

		fname_abs = os.path.join(self.dir, fname + '.json')

		d_copy = deepcopy(d)
		if 'train_class' in d_copy.keys():
			del d_copy['train_class']

		with open(fname_abs, 'w+') as f:
			json.dump(d_copy, f, indent=4)


	def setup_NN(self):

		'''
		Set up the NNs. The target NNs are never updated with GD, so we set
		requires_grad=False for them.
		'''

		self.NN_actor = DDPG_actor(self.N_state_terms, self.hidden_size, self.N_actions)
		self.NN_actor_target = DDPG_actor(self.N_state_terms, self.hidden_size, self.N_actions)

		self.NN_critic = DDPG_critic(self.N_state_terms, self.hidden_size, self.N_actions)
		self.NN_critic_target = DDPG_critic(self.N_state_terms, self.hidden_size, self.N_actions)

		opt_dict = {
			'adam' : optim.Adam,
			'Adam' : optim.Adam,
			'RMSprop' : optim.RMSprop,
			'RMSProp' : optim.RMSprop,
		}
		assert self.optim in opt_dict.keys()

		self.optimizer_actor = opt_dict[self.optim](self.NN_actor.parameters(), lr=self.LR_actor)
		self.optimizer_critic = opt_dict[self.optim](self.NN_critic.parameters(), lr=self.LR_critic)

		self.NN_actor_target.load_state_dict(self.NN_actor.state_dict())
		self.NN_critic_target.load_state_dict(self.NN_critic.state_dict())

		for l in self.NN_actor_target.layers:
			l.weight.requires_grad = False
			l.bias.requires_grad = False

		for l in self.NN_actor_target.layers:
			l.weight.requires_grad = False
			l.bias.requires_grad = False



	def save_model(self, **kwargs):

		model_save_dir = os.path.join(self.dir, 'saved_models')

		if os.path.exists(model_save_dir):
			shutil.rmtree(model_save_dir)

		os.mkdir(model_save_dir)

		NN_dict = {
			'NN_actor' : self.NN_actor,
			'NN_actor_target' : self.NN_actor_target,
			'NN_critic' : self.NN_critic,
			'NN_critic_target' : self.NN_critic_target
		}
		for name, NN in NN_dict.items():
			fname = os.path.join(model_save_dir, f'{name}.model')
			torch.save(NN.state_dict(), fname)


	def load_model(self):

		model_save_dir = os.path.join(self.dir, 'saved_models')
		assert os.path.exists(model_save_dir), 'model dir must exist!'

		NN_dict = {
			'NN_actor' : self.NN_actor,
			'NN_actor_target' : self.NN_actor_target,
			'NN_critic' : self.NN_critic,
			'NN_critic_target' : self.NN_critic_target
		}
		for name, NN in NN_dict.items():
			fname = os.path.join(model_save_dir, f'{name}.model')
			NN.load_state_dict(torch.load(fname))


	def update_target_NN(self):

		'''
		For soft updating the actor and critic NNs.
		'''


		for l, l_target in zip(self.NN_actor.layers, self.NN_actor_target.layers):
			l_target.weight.data = (1 - self.tau)*l_target.weight.data + self.tau*l.weight.data
			l_target.bias.data = (1 - self.tau)*l_target.bias.data + self.tau*l.bias.data

		for l, l_target in zip(self.NN_critic.layers, self.NN_critic_target.layers):
			l_target.weight.data = (1 - self.tau)*l_target.weight.data + self.tau*l.weight.data
			l_target.bias.data = (1 - self.tau)*l_target.bias.data + self.tau*l.bias.data


	@path_utils.timer
	def train(self, N_eps, N_steps, **kwargs):
		'''
		plot_train() plots:
			-self.R_batches (mean amortized reward per batch)
			-self.Q_loss_batches, pol_loss_batches, H_loss_batches (mean losses per batch)
			-self.R_train_hist (summed, unnormalized rewards of each episode, over whole train session)

		'''

		print('\nBeginning training...\n')

		if kwargs.get('reset_hist', True):
			self.R_batches = []
			self.policy_loss_batches = []
			self.Q_loss_batches = []
			self.H_loss_batches = []
			self.noise_sigma_hist = []
			self.tot_loss_batches = []
			self.episode_ending = None

			self.R_train_hist = []

			self.batch_saves = {}
			self.total_step = 0
			self.train_epochs = []

			R_avg = 0

		self.train_epochs.append(self.total_step)

		self.decay_noise_factor = (self.noise_decay_limit)**(1.0/N_eps)

		for ep in range(N_eps):
			self.episode(N_steps)

			if self.decay_noise:
				self.noise_sigma *= self.decay_noise_factor
				self.noise_sigma_hist.append(self.noise_sigma)

			R_tot = sum(self.last_ep_R_hist)
			if ep==0:
				R_avg = R_tot
			else:
				R_avg = 0.99*R_avg + 0.01*R_tot

			if ep % max(N_eps // 100, 1) == 0:
				#self.plot_episode(show_plot=False, save_plot=True, iter=self.total_step)
				#self.plot_VF(show_plot=False, save_plot=True, iter=self.total_step)
				#self.plot_policy(show_plot=False, save_plot=True, iter=self.total_step)
				#self.plot_trajectory(show_plot=False, save_plot=True, iter=self.total_step)
				plot_tools.save_traj_to_file(self.last_ep_state_hist,
												self.dir, iter=self.total_step,
												cat_speed_rel=self.agent.cat_speed_rel,
												mouse_caught=self.episode_ending)
				print('\nepisode {}/{}'.format(ep, N_eps))
				#print(f'ER buffer length = {self.ER.buf_len}')
				print(f'Episode R_tot = {R_tot:.4f}\t avg reward = {R_avg:.3f}')
				print(f'Buffer write index = {self.ER.write_index}, buffer len = {self.ER.buf_len}')

			if ep % max(N_eps // 100, 1) == 0:
					self.plot_train(save_plot=True, show_plot=False)

			self.total_step += 1



	def episode(self, N_steps):

		self.agent.initEpisode()

		'''

		The GD is done with:
			-v_buffer (batch of V for each state)
			-sigma_buffer (batch of sd for each state)
			-r_buffer (batch of returned r for each action taken in each state)
			-pi_a_buffer (batch of pi_a for each action taken in each state)

		plot_episode() plots:
			-self.last_ep_*_hist (history of * over course of episode)

		other buffers saved but not used (for debugging):
			-s_buffer
			-a_buffer
			-mu_buffer

		'''


		s_ep_hist = []
		a_ep_hist = []
		R_ep_hist = []
		noise_ep_hist = []
		self.episode_ending = None

		if self.noise_mu == 'zero':
			OU_noise = OrnsteinUhlenbeckNoise(np.array([0,0]),
													sigma=self.noise_sigma,
													theta=self.noise_theta,
													dt=self.noise_dt)
		else:
			OU_noise = OrnsteinUhlenbeckNoise(np.random.randn(2),
													sigma=self.noise_sigma,
													theta=self.noise_theta,
													dt=self.noise_dt)

		for t in range(N_steps):

			# DDPG stuff
			s = self.agent.getStateVec()

			if self.noise_method == 'OU':
				noise = OU_noise() # add noise?
			else:
				noise = self.noise_sigma*np.random.randn(2)


			# Get action from actor NN
			a = self.NN_actor.forward(torch.tensor(s, dtype=torch.float)).detach().numpy() + noise

			# Iterate, get r, s_next
			r, s_next, done = self.agent.iterate(a)
			# Add to ER buffer
			self.ER.add_exp(s, a, r, s_next, int(done))


			# Ep hist stuff
			R_ep_hist.append(r)
			s_ep_hist.append(s)
			a_ep_hist.append(a)
			noise_ep_hist.append(noise)


			if self.ER.buf_len >= 2*self.ER.batch_size:

				# Get a batch
				s_b, a_b, r_b, s_next_b, d_b = self.ER.get_batch()
				# Non terminal state variable
				non_term_b = 1.0 - d_b
				# Get the Q value for the (s,a) from NON target NN
				Q_b = self.NN_critic.forward(s_b, a_b).squeeze()
				# The a_next and Q_next are both calculated with target NNs
				a_next_b = self.NN_actor_target.forward(s_next_b)
				Q_next_b = self.NN_critic_target.forward(s_next_b, a_next_b).squeeze()

				# Calculate. Non-term handles whether it was terminal state or not
				adv = r_b + self.gamma*non_term_b*Q_next_b - Q_b

				# Zero both grads
				self.optimizer_actor.zero_grad()
				self.optimizer_critic.zero_grad()

				# Calculate Q loss, backprop, step. This should affect
				# ONLY the non-target critic NN.
				Q_loss = adv.pow(2).mean()
				Q_loss.backward()
				self.optimizer_critic.step()

				# The action to evaluate Q at is determined with the current
				# actor NN, and plugged into the critic NN. Then, it is backprop'd
				# with respect to the ACTOR NN.
				a_current_b = self.NN_actor.forward(s_b)
				policy_loss = -self.NN_critic.forward(s_b, a_current_b).mean()
				policy_loss.backward()
				self.optimizer_actor.step()


				# Append batches
				self.policy_loss_batches.append(policy_loss.detach().mean())
				self.Q_loss_batches.append(Q_loss.detach().mean())

				# Update target NNs incrementally
				self.update_target_NN()


			if done:
				s_ep_hist.append(s_next)
				break


		last_r = R_ep_hist[-1]
		if last_r > 0.5:
			self.episode_ending = 'escaped'
		elif last_r < -0.2:
			self.episode_ending = 'caught'
		else:
			self.episode_ending = None

		self.R_train_hist.append(sum(R_ep_hist))

		self.last_ep_R_hist = np.array(R_ep_hist)
		self.last_ep_state_hist = np.array(s_ep_hist)
		self.last_ep_action_hist = np.array(a_ep_hist)
		self.last_ep_noise_hist = np.array(noise_ep_hist)


	def no_train_episode(self, N_steps):

		self.agent.initEpisode()

		'''

		The GD is done with:
			-v_buffer (batch of V for each state)
			-sigma_buffer (batch of sd for each state)
			-r_buffer (batch of returned r for each action taken in each state)
			-pi_a_buffer (batch of pi_a for each action taken in each state)

		plot_episode() plots:
			-self.last_ep_*_hist (history of * over course of episode)

		other buffers saved but not used (for debugging):
			-s_buffer
			-a_buffer
			-mu_buffer

		'''


		s_ep_hist = []
		R_ep_hist = []
		episode_ending = None
		tot_steps = 0

		for t in range(N_steps):

			# DDPG stuff
			s = self.agent.getStateVec()

			# Get action from actor NN
			a = self.NN_actor.forward(torch.tensor(s, dtype=torch.float)).detach().numpy()

			# Iterate, get r, s_next
			r, s_next, done = self.agent.iterate(a)

			s_ep_hist.append(s)
			R_ep_hist.append(r)

			tot_steps = t
			if done:
				s_ep_hist.append(s_next)
				break


		last_r = R_ep_hist[-1]
		if last_r > 0.5:
			episode_ending = 'escaped'
		elif last_r < -0.2:
			episode_ending = 'caught'
		else:
			episode_ending = None

		traj_fname = plot_tools.save_traj_to_file(np.array(s_ep_hist),
										self.dir, iter='eval_ep_{}'.format(path_utils.get_date_str()),
										cat_speed_rel=self.agent.cat_speed_rel,
										mouse_caught=episode_ending)


		return {
				'traj_fname': traj_fname,
				'mouse_caught' : episode_ending,
				'tot_steps' : tot_steps
			}


	################################# Plotting functions

	def plot_VF(self, **kwargs):

		'''
		For plotting the value function (VF) as a function of the mouse's pos,
		for a fixed cat pos.
		'''

		VF_dir = os.path.join(self.dir, 'VF_plots')
		if not os.path.exists(VF_dir):
			os.mkdir(VF_dir)


		plt.close('all')

		plt.figure(figsize=(8,8))
		ax = plt.gca()

		N_grid = 30


		cat_pos = kwargs.get('cat_pos', np.array([1.0,0.0]))

		cat_pos = np.array(cat_pos)*1.07

		grid_x = np.linspace(-1.0, 1.0, N_grid)
		grid_y = np.linspace(-1.0, 1.0, N_grid)

		# The coords go like a matrix, so the first one is the y direction
		V = np.zeros((N_grid, N_grid))

		dat = []

		for i,x in enumerate(grid_x):
			for j,y in enumerate(grid_y):
				s = torch.tensor([x, y, *cat_pos], dtype=torch.float).unsqueeze(dim=0)
				out_pi = self.NN_actor.forward(s).detach()
				out_v = self.NN_critic.forward(s, out_pi).detach()
				dat.append([x, y, out_v.item()])

		dat = np.array(dat)
		x = dat[:,0]
		y = dat[:,1]
		z = dat[:,2]

		x = np.unique(x)
		y = np.unique(y)
		X,Y = np.meshgrid(x,y)

		Z=z.reshape(len(y),len(x))

		plt.pcolormesh(X,Y,Z, cmap='coolwarm')
		plt.colorbar(fraction=0.046, pad=0.04)

		circle_main = plt.Circle((0, 0), 1.0, edgecolor='black', lw=1.5, fill=False)
		ax.add_artist(circle_main)

		circle_cat = plt.Circle(cat_pos, 0.05, edgecolor='black', lw=0.7, fill=False)
		ax.add_artist(circle_cat)
		ax.set_aspect('equal')

		plt.xlim(-1.2, 1.2)
		plt.ylim(-1.2, 1.2)

		plt.tight_layout()

		'''
		cax = ax.matshow(V)
		cbar = plt.colorbar(cax, fraction=0.046, pad=0.04)
		plt.xlabel('x', fontsize=16)
		plt.ylabel('y', fontsize=16)'''
		'''ax.set_xticks([0, N_grid_thetadot-1])
		ax.set_xticklabels(['-8', '8'])
		ax.set_yticks([0, N_grid_theta-1])
		ax.set_yticklabels(['-pi', 'pi'])'''

		if kwargs.get('save_plot', True):
			if kwargs.get('iter', None) is not None:
				plt.savefig(os.path.join(VF_dir, '{}_VF_{}.png'.format(self.fname_base, kwargs.get('iter'))))
			else:
				plt.savefig(os.path.join(VF_dir, '{}_catpos={}_VF.png'.format(self.fname_base, cat_pos)))
		if kwargs.get('show_plot', False):
			plt.show()

		plt.close('all')


	def plot_policy(self, **kwargs):

		'''
		For plotting the policy as a function of the mouse's pos,
		for a fixed cat pos.
		'''

		policy_dir = os.path.join(self.dir, 'policy_plots')
		if not os.path.exists(policy_dir):
			os.mkdir(policy_dir)

		plt.close('all')

		plt.figure(figsize=(8,8))
		ax = plt.gca()

		N_grid = 30


		cat_pos = kwargs.get('cat_pos', np.array([1.0,0.0]))

		cat_pos = np.array(cat_pos)*1.07

		grid_x = np.linspace(-1.0, 1.0, N_grid)
		grid_y = np.linspace(-1.0, 1.0, N_grid)

		# The coords go like a matrix, so the first one is the y direction
		V = np.zeros((N_grid, N_grid))

		grid_sq_size = 2.0/N_grid
		arrow_len = grid_sq_size/2.0
		w = 0.004

		dat = []
		for i,x in enumerate(grid_x):
			for j,y in enumerate(grid_y):
				s = torch.tensor([x, y, *cat_pos], dtype=torch.float)
				out_pi = self.NN_actor.forward(s)
				dx, dy = out_pi.detach().numpy()
				move_ang = np.arctan2(dy, dx)

				ax.arrow(x, y,
						arrow_len*np.cos(move_ang), arrow_len*np.sin(move_ang),
								width=w,
								head_width=5*w,
								color='black')

		circle_main = plt.Circle((0, 0), 1.0, edgecolor='black', lw=1.5, fill=False)
		ax.add_artist(circle_main)

		circle_cat = plt.Circle(cat_pos, 0.05, edgecolor='black', lw=0.7, fill=False)
		ax.add_artist(circle_cat)
		ax.set_aspect('equal')

		plt.xlim(-1.2, 1.2)
		plt.ylim(-1.2, 1.2)

		plt.tight_layout()

		if kwargs.get('save_plot', True):
			if kwargs.get('iter', None) is not None:
				plt.savefig(os.path.join(policy_dir, '{}_policy_{}.png'.format(self.fname_base, kwargs.get('iter'))))
			else:
				plt.savefig(os.path.join(policy_dir, '{}_catpos={}_policy.png'.format(self.fname_base, cat_pos)))
		if kwargs.get('show_plot', False):
			plt.show()

		plt.close('all')


	def plot_episode(self, **kwargs):

		'''
		For plotting the stats/state variables of the cat and, mouse, and
		more, during an ep.
		'''

		ep_dir = os.path.join(self.dir, 'ep_plots')

		if not os.path.exists(ep_dir):
			os.mkdir(ep_dir)


		plt.close('all')

		fig, axes = plt.subplots(2, 4, figsize=(12,6))

		ax_mouse_cos = axes[0][0]
		ax_mouse_sin = axes[0][1]

		ax_mouse_mu_x = axes[0][2]
		ax_mouse_mu_y = axes[0][3]

		ax_cat_cos = axes[1][0]
		ax_cat_sin = axes[1][1]

		ax_R = axes[1][2]
		ax_noise = axes[1][3]




		mouse_pos = self.last_ep_state_hist[:,:2]
		cat_pos = self.last_ep_state_hist[:,2:]

		ax_mouse_cos.set_xlabel('step')
		ax_mouse_cos.set_ylabel('mouse x')
		ax_mouse_cos.plot(mouse_pos[:,0], color='dodgerblue')
		ax_mouse_cos.set_ylim(-1.1, 1.1)

		ax_mouse_sin.set_xlabel('step')
		ax_mouse_sin.set_ylabel('mouse y')
		ax_mouse_sin.plot(mouse_pos[:,1], color='tomato')
		ax_mouse_sin.set_ylim(-1.1, 1.1)


		ax_cat_cos.set_xlabel('step')
		ax_cat_cos.set_ylabel('cat x')
		ax_cat_cos.plot(cat_pos[:,0], color='forestgreen')
		ax_cat_cos.set_ylim(-1.1, 1.1)

		ax_cat_sin.set_xlabel('step')
		ax_cat_sin.set_ylabel('cat y')
		ax_cat_sin.plot(cat_pos[:,1], color='darkorange')
		ax_cat_sin.set_ylim(-1.1, 1.1)

		#step_vals = list(range(len(self.last_ep_mu_hist)))

		ax_mouse_mu_x.set_xlabel('step')
		ax_mouse_mu_x.set_ylabel('mouse mu_x')
		ax_mouse_mu_x.plot(self.last_ep_action_hist[:,0], '.', color='black', markersize=2)
		'''ax_mouse_mu_x.fill_between(step_vals,
									self.last_ep_mu_hist[:,0] - self.last_ep_sigma_hist[:,0],
									self.last_ep_mu_hist[:,0] + self.last_ep_sigma_hist[:,0],
										color='dodgerblue', alpha=0.3)'''
		#ax_mouse_mu_x.plot(self.last_ep_mu_hist[:,0], color='darkblue')



		ax_mouse_mu_y.set_ylabel('step')
		ax_mouse_mu_y.set_ylabel('mouse mu_y')
		ax_mouse_mu_y.plot(self.last_ep_action_hist[:,1], '.', color='black', markersize=2)
		'''ax_mouse_mu_y.fill_between(step_vals,
									self.last_ep_mu_hist[:,1] - self.last_ep_sigma_hist[:,1],
									self.last_ep_mu_hist[:,1] + self.last_ep_sigma_hist[:,1],
										color='tomato', alpha=0.3)'''
		#ax_mouse_mu_y.plot(self.last_ep_mu_hist[:,1], color='darkred')


		ax_R.set_xlabel('step')
		ax_R.set_ylabel('R')
		ax_R.plot(self.last_ep_R_hist, color='forestgreen')


		ax_noise.set_xlabel('step')
		ax_noise.set_ylabel('noise')
		ax_noise.plot(self.last_ep_noise_hist[:,0], color='dodgerblue')
		ax_noise.plot(self.last_ep_noise_hist[:,1], color='darkorange')


		plt.tight_layout()

		if kwargs.get('save_plot', True):
			if kwargs.get('iter', None) is not None:
				plt.savefig(os.path.join(ep_dir, 'ep_{}.png'.format(kwargs.get('iter'))))
			else:
				plt.savefig(os.path.join(ep_dir, 'ep.png'))
		if kwargs.get('show_plot', True):
			plt.show()


	def plot_train(self, **kwargs):

		'''
		For plotting training stats, including average reward, batch info,
		losses, etc.
		'''


		plt.close('all')

		fig, axes = plt.subplots(2, 2, figsize=(10,8))

		ax_R = axes[0][0]
		ax_noise_sigma = axes[0][1]
		ax_Q_loss = axes[1][0]
		ax_policy_loss = axes[1][1]

		for ep in self.train_epochs:
			ax_R.axvline(ep, linestyle='dashed', alpha=0.8, color='gray')

		ax_R.set_xlabel('Episode')
		ax_R.set_ylabel('R')
		ax_R.plot(self.R_train_hist, color='dodgerblue', alpha=0.5)
		ax_R.plot(*plot_tools.smooth_data(self.R_train_hist), color='black')

		ax_Q_loss.set_xlabel('Batch')
		ax_Q_loss.set_ylabel('Q loss')
		ax_Q_loss.plot(self.Q_loss_batches, color='tomato')

		ax_noise_sigma.set_xlabel('Episode')
		ax_noise_sigma.set_ylabel('Action noise')
		ax_noise_sigma.plot(self.noise_sigma_hist, color='forestgreen')

		ax_policy_loss.set_xlabel('Batch')
		ax_policy_loss.set_ylabel('Policy loss')
		ax_policy_loss.plot(self.policy_loss_batches, color='darkorange')

		'''ax_tot_loss.set_xlabel('batch')
		ax_tot_loss.set_ylabel('loss_tot')
		ax_tot_loss.plot(self.tot_loss_batches, color='mediumseagreen')'''

		plt.suptitle(self.train_plot_title, fontsize=10)

		plt.tight_layout()
		fig.subplots_adjust(top=0.85)

		if kwargs.get('save_plot', True):
			if kwargs.get('iter', None) is not None:
				plt.savefig(os.path.join(self.dir, '{}_train_{}.png'.format(self.fname_base, kwargs.get('iter'))))
			else:
				plt.savefig(os.path.join(self.dir, '{}_train.png'.format(self.fname_base)))
		if kwargs.get('show_plot', True):
			plt.show()




##################### ER buffers and NN stuff



class ExpReplay:

	def __init__(self, buffer_max_size, batch_size):

		self.buffer_max_size = buffer_max_size
		self.batch_size = batch_size

		self.reset_buffer()


	def reset_buffer(self):
		self.buffer = []

		self.write_index = 0
		self.buf_len = 0


	def get_batch(self):
		batch = random.sample(self.buffer, min(self.batch_size, self.buf_len))

		s_b = torch.tensor([b.s for b in batch], dtype=torch.float)
		a_b = torch.tensor([b.a for b in batch], dtype=torch.float)
		r_b = torch.tensor([b.r for b in batch], dtype=torch.float)
		s_next_b = torch.tensor([b.s_next for b in batch], dtype=torch.float)
		d_b = torch.tensor([b.d for b in batch], dtype=torch.float)

		return(s_b, a_b, r_b, s_next_b, d_b)


	def add_exp(self, s, a, r, s_next, d):

		# Either append an experience to the buffer,
		# or overwrite an old one.
		if self.buf_len < self.buffer_max_size:
			self.buffer.append(Experience(s, a, r, s_next, d))
			self.buf_len += 1
		else:
			self.buffer[self.write_index] = Experience(s, a, r, s_next, d)
			self.write_index = (self.write_index + 1) % self.buffer_max_size


class Experience:

	def __init__(self, s, a, r, s_next, d):
		# d is whether it's a terminal state (done)
		self.s = s
		self.a = a
		self.r = r
		self.s_next = s_next
		self.d = d



class DDPG_actor(nn.Module):

	def __init__(self, N_state_inputs, H, N_action_outputs, **kwargs):
		super(DDPG_actor, self).__init__()

		self.actor_lin1 = nn.Linear(N_state_inputs, H)
		self.actor_lin2 = nn.Linear(H, H)
		self.actor_out = nn.Linear(H, N_action_outputs)

		fan_in_1 = np.sqrt(1/N_state_inputs)
		nn.init.uniform_(self.actor_lin1.weight, -fan_in_1, fan_in_1)
		nn.init.uniform_(self.actor_lin1.bias, -fan_in_1, fan_in_1)
		fan_in_2 = np.sqrt(1/H)
		nn.init.uniform_(self.actor_lin2.weight, -fan_in_2, fan_in_2)
		nn.init.uniform_(self.actor_lin2.bias, -fan_in_2, fan_in_2)
		last_layer_init = 3*10**-3
		nn.init.uniform_(self.actor_out.weight, -last_layer_init, last_layer_init)
		nn.init.uniform_(self.actor_out.bias, -last_layer_init, last_layer_init)

		self.nonlinear = nn.ReLU()
		self.t = nn.Tanh()

		self.layers = [
						self.actor_lin1,
						self.actor_lin2,
						self.actor_out
					]

	def forward(self, x_state):

		y = self.actor_lin1(x_state)
		y = self.nonlinear(y)
		y = self.actor_lin2(y)
		y = self.nonlinear(y)
		y = self.actor_out(y)
		y = self.t(y)

		return(y)


class DDPG_critic(nn.Module):

	def __init__(self, N_state_inputs, H, N_action_inputs, **kwargs):
		super(DDPG_critic, self).__init__()

		self.critic_lin1 = nn.Linear(N_state_inputs, H)
		self.critic_lin2 = nn.Linear(H + N_action_inputs, H)
		self.critic_out = nn.Linear(H, 1)
		self.nonlinear = nn.ReLU()

		fan_in_1 = np.sqrt(1/N_state_inputs)
		nn.init.uniform_(self.critic_lin1.weight, -fan_in_1, fan_in_1)
		nn.init.uniform_(self.critic_lin1.bias, -fan_in_1, fan_in_1)

		fan_in_2 = np.sqrt(1/(H + N_action_inputs))
		nn.init.uniform_(self.critic_lin2.weight, -fan_in_2, fan_in_2)
		nn.init.uniform_(self.critic_lin2.bias, -fan_in_2, fan_in_2)

		last_layer_init = 3*10**-3
		nn.init.uniform_(self.critic_out.weight, -last_layer_init, last_layer_init)
		nn.init.uniform_(self.critic_out.bias, -last_layer_init, last_layer_init)


		self.layers = [	self.critic_lin1,
						self.critic_lin2,
						self.critic_out
						]


	def forward(self, x_state, x_action):


		y = self.critic_lin1(x_state)
		y = self.nonlinear(y)
		z = torch.cat((y, x_action), dim=1)

		z = self.critic_lin2(z)
		z = self.nonlinear(z)
		z = self.critic_out(z)

		return(z)



#
