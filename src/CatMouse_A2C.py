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
import os, json, random
import traceback as tb
from copy import deepcopy

# Ones specific to this project
from CatMouseAgent import CatMouseAgent
from OrnsteinUhlenbeckNoise import OrnsteinUhlenbeckNoise
import plot_tools

'''

Pretty straightforward implementation of an Actor Critic type
architecture. Updates with batches during episode (i.e., every 5
iterations of an ep, it updates).

Continuous action output: meaning, it outputs mu and sigma, and
you draw from that distribution.

This is pretty similar to A2C.

'''



default_kwargs = {
	'N_batch' : 5,
	'gamma' : 0.99,
	'beta_entropy' : 0.005,
	'optim' : 'RMSprop',
	'LR' : 2*10**-4,
	'clamp_grad' : 10000.0,
	'hidden_size' : 200,
	'base_dir' : path_utils.get_output_dir(),
	'decay_noise' : True,
	'noise_sigma' : 0.2,
	'noise_theta' : 0.1,
	'noise_dt' : 10**-2,
	'sigma_min' : 10**-3

}

class CatMouse_A2C:


	def __init__(self, **kwargs):
		global default_kwargs

		self.agent = CatMouseAgent(**kwargs)

		self.N_state_terms = len(self.agent.getStateVec())
		self.N_actions = self.agent.N_actions

		for k,v in kwargs.items():
			if k in default_kwargs.keys():
				default_kwargs[k] = v
			else:
				print(f'Passed parameter {k} not in default_kwargs dict! Check')
				if k != 'fname_note':
					#assert k in default_kwargs.keys(), 'key error'
					pass

		self.fname_base = 'CatMouse_A2C_' + path_utils.get_date_str()

		self.dir = os.path.join(default_kwargs['base_dir'], self.fname_base)
		os.mkdir(self.dir)
		print(f'\n\nMade dir {self.dir} for run...\n\n')

		self.save_params_json(kwargs, 'params_passed')
		self.save_params_json(default_kwargs, 'params_all')

		self.N_batch = default_kwargs['N_batch']
		self.gamma = default_kwargs['gamma']
		self.beta_entropy = default_kwargs['beta_entropy']
		self.optim = default_kwargs['optim']
		self.LR = default_kwargs['LR']
		self.hidden_size = default_kwargs['hidden_size']
		self.clamp_grad = default_kwargs['clamp_grad']
		self.noise_sigma = default_kwargs['noise_sigma']
		self.noise_theta = default_kwargs['noise_theta']
		self.noise_dt = default_kwargs['noise_dt']
		self.sigma_min = default_kwargs['sigma_min']
		self.decay_noise = default_kwargs['decay_noise']


		self.setup_NN()
		self.dat_fname_file = None

		if 'base_dir' in default_kwargs.keys():
			no_base_dir_kwargs = deepcopy(default_kwargs)
			del no_base_dir_kwargs['base_dir'] # To get rid of this for the center dict
		self.train_plot_title = path_utils.linebreak_every_n_spaces(path_utils.param_dict_to_label_str(no_base_dir_kwargs))


	def __repr__(self):
		return 'CatMouse_A2C object'

	def save_params_json(self, d, fname):

		fname_abs = os.path.join(self.dir, fname + '.json')

		d_copy = deepcopy(d)
		if 'train_class' in d_copy.keys():
			del d_copy['train_class']

		with open(fname_abs, 'w+') as f:
			json.dump(d_copy, f, indent=4)


	def setup_NN(self):

		self.NN = AC_NN(self.N_state_terms,
						self.hidden_size,
						self.N_actions,
						sigma_min=self.sigma_min)

		for l in self.NN.layers:
			nn.init.normal_(l.weight, mean=0.0, std=0.1)
			#nn.init.normal_(l.bias, mean=0., std=0.1)
			nn.init.constant_(l.bias, 0.0)

		if self.optim == 'Adam':
			self.optimizer = optim.Adam(self.NN.parameters(), lr=self.LR)
		else:
			self.optimizer = optim.RMSprop(self.NN.parameters(), lr=self.LR)


	@path_utils.timer
	def train(self, N_eps, N_steps, **kwargs):
		'''
		plot_train() plots:
			-self.R_batches (mean amortized reward per batch)
			-self.V_loss_batches, pol_loss_batches, H_loss_batches (mean losses per batch)
			-self.R_train_hist (summed, unnormalized rewards of each episode, over whole train session)

		'''

		print('\nBeginning training...\n')

		self.R_batches = []
		self.pol_loss_batches = []
		self.V_loss_batches = []
		self.H_loss_batches = []
		self.tot_loss_batches = []

		self.R_train_hist = []
		self.noise_sigma_hist = []
		self.batch_saves = {}
		self.global_step = 0
		self.total_step = 0

		R_avg = 0

		self.decay_noise_factor = (10**-3)**(1.0/N_eps)


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
				plot_tools.save_traj_to_file(self.last_ep_state_hist,
												self.dir, iter=self.total_step,
												cat_speed_rel=self.agent.cat_speed_rel)
				#self.plot_episode(show_plot=False, save_plot=True, iter=ep)
				#self.plot_VF(show_plot=False, save_plot=True, iter=ep)
				#self.plot_trajectory(show_plot=False, save_plot=True, iter=ep)
				print('\nepisode {}/{}'.format(ep, N_eps))
				print(f'Episode R_tot = {R_tot:.4f}\t avg reward = {R_avg:.1f}')

			if ep % max(N_eps // 1000, 1) == 0:
					self.plot_train(save_plot=True, show_plot=False)

			self.total_step += 1

		# Do it at the end again, just to have a last one
		self.plot_train(save_plot=True, show_plot=False)


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
		sigma_ep_hist = []
		mu_ep_hist = []
		a_ep_hist = []
		R_ep_hist = []
		noise_ep_hist = []

		# Needed for calculating loss
		v_buffer = []
		r_buffer = []
		sigma_buffer = []
		pi_a_buffer = []

		# Others
		s_buffer = []
		a_buffer = []
		mu_buffer = []


		OU_noise = OrnsteinUhlenbeckNoise(np.array([0,0]),
												sigma=self.noise_sigma,
												theta=self.noise_theta,
												dt=self.noise_dt)

		for t in range(N_steps):

			s = self.agent.getStateVec()

			out_V, out_pi = self.NN.forward(torch.tensor(s, dtype=torch.float))
			mu, sigma = out_pi

			mu = mu.squeeze()
			sigma = sigma.squeeze()
			# Calculate the dist. we're gonna draw from.

			pi_dist = Normal(mu, sigma)
			noise = OU_noise()
			a_tens = pi_dist.sample() + torch.tensor(noise, dtype=torch.float)

			a = a_tens.tolist()
			pi_val = pi_dist.log_prob(a_tens).sum()

			r, s_next, done = self.agent.iterate(a)
			R_ep_hist.append(r)


			# Needed to calc loss
			sigma_buffer.append(torch.log(sigma).sum())
			v_buffer.append(out_V)
			pi_a_buffer.append(pi_val)
			r_buffer.append(r)

			# Others, debugging
			s_buffer.append(s.tolist())
			a_buffer.append(a)
			mu_buffer.append(mu.detach().tolist())

			# Ep hist stuff
			s_ep_hist.append(s)
			a_ep_hist.append(a)
			noise_ep_hist.append(noise)
			sigma_ep_hist.append(sigma.detach().tolist())
			mu_ep_hist.append(mu.detach().tolist())

			if done or (t + 1)%self.N_batch==0:
				if not done:
					# If the ep wasnt done, add V for s_next to the last R received.
					out_V, out_pi = self.NN.forward(torch.tensor(s_next, dtype=torch.float))
					r_buffer[-1] += self.gamma*out_V.detach().item()

				buf_len = len(r_buffer)

				discount_gamma = torch.tensor(
					np.array([[self.gamma**(c-r) if c>=r else 0 for c in range(buf_len)] for r in range(buf_len)])
					, dtype=torch.float)

				# Calculated decayed R over batch
				R = torch.tensor(r_buffer, dtype=torch.float).unsqueeze(dim=1)
				r_batch_disc = torch.mm(discount_gamma, R).squeeze(dim=1)

				# Convert to tensors (be careful!)
				V_batch = torch.stack(v_buffer).squeeze(dim=1)
				pi_batch = torch.stack(pi_a_buffer)
				sigma_batch = torch.stack(sigma_buffer)

				adv = r_batch_disc - V_batch
				policy_loss = (-adv.detach()*pi_batch)
				V_loss = adv.pow(2)
				# I think
				H_loss = -self.beta_entropy*(0.5 + 0.5*np.log(2*3.14) + sigma_batch)

				loss_tot = (policy_loss + V_loss + H_loss).mean()

				self.optimizer.zero_grad()
				loss_tot.backward()

				grad_norm_thresh = 500.0

				total_norm = 0
				for p in self.NN.parameters():
					param_norm = p.grad.data.norm(2.0)
					total_norm += param_norm.item() ** 2.0

				total_norm = total_norm ** (1. / 2.0)
				#self.save_dat_append('grad_norm', np.array([total_norm]))

				if total_norm > grad_norm_thresh:
					print(f'Grad norm ({total_norm:.2f} above threshold ({grad_norm_thresh}))')
					'''
					self.batch_saves[self.global_step] = {
													'grad_norm' : total_norm,
													'r' : r_buffer,
													'r_batch' : r_batch_disc.detach().tolist(),
													'V_batch' : V_batch.detach().tolist(),
													'pi_batch' : pi_batch.detach().tolist(),
													'sigma_batch' : sigma_batch.detach().tolist(),
													'mu_batch' : mu_buffer,
													's_batch' : s_buffer,
													'adv' : adv.detach().tolist(),
													'policy_loss' : policy_loss.detach().tolist(),
													'V_loss' : V_loss.detach().tolist(),
													'H_loss' : H_loss.detach().tolist(),

					}'''
					pass

				# Clamp gradient
				for l in self.NN.layers:
					l.weight.grad.data.clamp_(-self.clamp_grad, self.clamp_grad)
					l.bias.grad.data.clamp_(-self.clamp_grad, self.clamp_grad)

				self.optimizer.step()

				# Append batches
				self.R_batches.append(r_batch_disc.detach().mean())
				self.pol_loss_batches.append(policy_loss.detach().mean())
				self.V_loss_batches.append(V_loss.detach().mean())
				self.H_loss_batches.append(H_loss.detach().mean())
				self.tot_loss_batches.append(loss_tot.detach().item())

				self.global_step += 1

				# Reset buffers
				s_buffer = []
				a_buffer = []
				sigma_buffer = []
				mu_buffer = []
				r_buffer = []
				v_buffer = []
				pi_a_buffer = []


			if done:
				break


		self.R_train_hist.append(sum(R_ep_hist))

		self.last_ep_R_hist = np.array(R_ep_hist)
		self.last_ep_state_hist = np.array(s_ep_hist)
		self.last_ep_mu_hist = np.array(mu_ep_hist)
		self.last_ep_sigma_hist = np.array(sigma_ep_hist)
		self.last_ep_action_hist = np.array(a_ep_hist)
		self.last_ep_noise_hist = np.array(noise_ep_hist)



	def plot_VF(self, **kwargs):

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
				s = torch.tensor([x, y, *cat_pos], dtype=torch.float)
				out_v, out_pi = self.NN.forward(s)
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
				plt.savefig(os.path.join(self.dir, '{}_VF_{}.png'.format(self.fname_base, kwargs.get('iter'))))
			else:
				plt.savefig(os.path.join(self.dir, '{}_catpos={}_VF.png'.format(self.fname_base, cat_pos)))
		if kwargs.get('show_plot', False):
			plt.show()

		plt.close('all')


		'''###################

		plt.close('all')
		s = 5
		plt.figure(figsize=(s,s), frameon=False)

		ax = plt.gca()
		ax.axis('off')
		plt.autoscale(tight=True)
		mouse_pos = self.last_ep_state_hist[:,:2]
		cat_pos = self.last_ep_state_hist[:,2:]
		#cat_angle = 3.14*(np.array(self.last_ep_state_hist[:,2]) + 1)
		#print(np.array(self.last_ep_state_hist[:,3]))
		#print(cat_angle)
		cat_rad = 1.1
		cat_pos_x = cat_pos[:,0]
		cat_pos_y = cat_pos[:,1]
		#cat_pos_x = cat_rad*np.cos(cat_angle)
		#cat_pos_y = cat_rad*np.sin(cat_angle)
		#print(mouse_pos)
		plt.plot(mouse_pos[:,0], mouse_pos[:,1], color='dodgerblue')

		circle_last_mouse_pos = plt.Circle((mouse_pos[-1,0], mouse_pos[-1,1]), 0.03, fill=False, edgecolor='mediumseagreen', linewidth=2)
		ax.add_artist(circle_last_mouse_pos)

		plt.scatter(cat_pos_x, cat_pos_y, s=200, edgecolor='black', linewidth=0.2, marker='o', facecolors='None')
		plt.scatter(cat_pos_x[-1], cat_pos_y[-1], s=120, edgecolor='red', linewidth=2.5, marker='o', facecolors='None')
		plt.scatter(cat_pos_x[0], cat_pos_y[0], s=120, edgecolor='blue', linewidth=2.5, marker='o', facecolors='None')
		#plt.plot(cat_pos_x, cat_pos_y, color='black', markersize=8, marker='o', fillstyle=None)


		circle_main = plt.Circle((0, 0), 1.0, edgecolor='tomato', fill=False)
		ax.add_artist(circle_main)


		r_dash = max(0, 1.0-3.14/self.agent.cat_speed_rel)
		circle_dash = plt.Circle((0, 0), r_dash, linestyle='dashed', edgecolor='grey', fill=False)
		ax.add_artist(circle_dash)

		r_match = 1.0/self.agent.cat_speed_rel
		circle_match = plt.Circle((0, 0), r_match, edgecolor='lightgray', fill=False)
		ax.add_artist(circle_match)


		lim = 1.3
		ax.set_xlim(-lim, lim)
		ax.set_ylim(-lim, lim)
		ax.set_aspect('equal')


		if kwargs.get('save_plot', True):
			if kwargs.get('iter', None) is not None:
				plt.savefig(os.path.join(traj_dir, 'traj_{}.png'.format(kwargs.get('iter'))))
			else:
				plt.savefig(os.path.join(traj_dir, 'traj.png'))
		if kwargs.get('show_plot', True):
			plt.show()


		plt.close('all')'''



	def plot_episode(self, **kwargs):

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

		step_vals = list(range(len(self.last_ep_mu_hist)))

		ax_mouse_mu_x.set_xlabel('step')
		ax_mouse_mu_x.set_ylabel('mouse mu_x')
		ax_mouse_mu_x.plot(self.last_ep_action_hist[:,0], '.', color='black', markersize=2)
		ax_mouse_mu_x.fill_between(step_vals,
									self.last_ep_mu_hist[:,0] - self.last_ep_sigma_hist[:,0],
									self.last_ep_mu_hist[:,0] + self.last_ep_sigma_hist[:,0],
										color='dodgerblue', alpha=0.3)
		ax_mouse_mu_x.plot(self.last_ep_mu_hist[:,0], color='darkblue')



		ax_mouse_mu_y.set_ylabel('step')
		ax_mouse_mu_y.set_ylabel('mouse mu_y')
		ax_mouse_mu_y.plot(self.last_ep_action_hist[:,1], '.', color='black', markersize=2)
		ax_mouse_mu_y.fill_between(step_vals,
									self.last_ep_mu_hist[:,1] - self.last_ep_sigma_hist[:,1],
									self.last_ep_mu_hist[:,1] + self.last_ep_sigma_hist[:,1],
										color='tomato', alpha=0.3)
		ax_mouse_mu_y.plot(self.last_ep_mu_hist[:,1], color='darkred')


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

		plt.close('all')

		fig, axes = plt.subplots(2, 2, figsize=(10,8))

		ax_R = axes[0][0]
		ax_H_loss = axes[0][1]
		ax_V_loss = axes[1][0]
		ax_policy_loss = axes[1][1]

		ax_R.set_xlabel('Episode')
		ax_R.set_ylabel('R')
		ax_R.plot(self.R_train_hist, color='dodgerblue', alpha=0.5)
		ax_R.plot(*plot_tools.smooth_data(self.R_train_hist), color='black')

		ax_V_loss.set_xlabel('Batch')
		ax_V_loss.set_ylabel('V loss')
		ax_V_loss.plot(self.V_loss_batches, color='tomato')

		ax_H_loss.set_xlabel('Batch')
		ax_H_loss.set_ylabel('Entropy loss')
		ax_H_loss.plot(self.H_loss_batches, color='forestgreen')

		ax_policy_loss.set_xlabel('Batch')
		ax_policy_loss.set_ylabel('Policy loss')
		ax_policy_loss.plot(self.pol_loss_batches, color='darkorange')

		'''ax_tot_loss.set_xlabel('Batch')
		ax_tot_loss.set_ylabel('Total loss')
		ax_tot_loss.plot(self.tot_loss_batches, color='mediumseagreen')'''

		plt.suptitle(self.train_plot_title, fontsize=10)

		plt.tight_layout()
		fig.subplots_adjust(top=0.8)

		if kwargs.get('save_plot', True):
			if kwargs.get('iter', None) is not None:
				plt.savefig(os.path.join(self.dir, '{}_train_{}.png'.format(self.fname_base, kwargs.get('iter'))))
			else:
				plt.savefig(os.path.join(self.dir, '{}_train.png'.format(self.fname_base)))
		if kwargs.get('show_plot', True):
			plt.show()



	def plot_trajectory(self, **kwargs):

		traj_dir = os.path.join(self.dir, 'traj_plots')

		if not os.path.exists(traj_dir):
			os.mkdir(traj_dir)



		plt.close('all')
		s = 5
		plt.figure(figsize=(s,s), frameon=False)

		ax = plt.gca()
		ax.axis('off')
		plt.autoscale(tight=True)
		mouse_pos = self.last_ep_state_hist[:,:2]
		cat_pos = self.last_ep_state_hist[:,2:]
		#cat_angle = 3.14*(np.array(self.last_ep_state_hist[:,2]) + 1)
		#print(np.array(self.last_ep_state_hist[:,3]))
		#print(cat_angle)
		cat_rad = 1.1
		cat_pos_x = cat_pos[:,0]
		cat_pos_y = cat_pos[:,1]
		#cat_pos_x = cat_rad*np.cos(cat_angle)
		#cat_pos_y = cat_rad*np.sin(cat_angle)
		#print(mouse_pos)
		plt.plot(mouse_pos[:,0], mouse_pos[:,1], color='dodgerblue')

		circle_last_mouse_pos = plt.Circle((mouse_pos[-1,0], mouse_pos[-1,1]), 0.03, fill=False, edgecolor='mediumseagreen', linewidth=2)
		ax.add_artist(circle_last_mouse_pos)

		plt.scatter(cat_pos_x, cat_pos_y, s=200, edgecolor='black', linewidth=0.2, marker='o', facecolors='None')
		plt.scatter(cat_pos_x[-1], cat_pos_y[-1], s=120, edgecolor='red', linewidth=2.5, marker='o', facecolors='None')
		plt.scatter(cat_pos_x[0], cat_pos_y[0], s=120, edgecolor='blue', linewidth=2.5, marker='o', facecolors='None')
		#plt.plot(cat_pos_x, cat_pos_y, color='black', markersize=8, marker='o', fillstyle=None)


		circle_main = plt.Circle((0, 0), 1.0, edgecolor='tomato', fill=False)
		ax.add_artist(circle_main)


		r_dash = max(0, 1.0-3.14/self.agent.cat_speed_rel)
		circle_dash = plt.Circle((0, 0), r_dash, linestyle='dashed', edgecolor='grey', fill=False)
		ax.add_artist(circle_dash)

		r_match = 1.0/self.agent.cat_speed_rel
		circle_match = plt.Circle((0, 0), r_match, edgecolor='lightgray', fill=False)
		ax.add_artist(circle_match)


		lim = 1.3
		ax.set_xlim(-lim, lim)
		ax.set_ylim(-lim, lim)
		ax.set_aspect('equal')

		plt.tight_layout()

		if kwargs.get('save_plot', True):
			if kwargs.get('iter', None) is not None:
				plt.savefig(os.path.join(traj_dir, 'traj_{}.png'.format(kwargs.get('iter'))))
			else:
				plt.savefig(os.path.join(traj_dir, 'traj.png'))
		if kwargs.get('show_plot', True):
			plt.show()


		plt.close('all')




	def save_dat_append(self, dat_name, dat):

		if self.dat_fname_file is None:
			self.dat_fname_file = os.path.join(self.dir, 'data_files.json')
			self.dat_fname_dict = {}
			print('Creating dat fname/shape dict...')

		if dat_name not in self.dat_fname_dict.keys():
			self.dat_fname_dict[dat_name] = {}

			print(f'Adding var {dat_name} to data dict...')
			print(f'Var has shape {dat.shape}')
			dat_fname = os.path.join(self.dir, '{}.dat'.format(dat_name))
			self.dat_fname_dict[dat_name]['fname'] = dat_fname
			self.dat_fname_dict[dat_name]['shape'] = dat.shape

			print(f'Creating new data dict file...')
			if os.path.exists(self.dat_fname_file):
				os.remove(self.dat_fname_file)

			with open(self.dat_fname_file, 'w+') as f:
				json.dump(self.dat_fname_dict, f, indent=4)

		dat_fname = self.dat_fname_dict[dat_name]['fname']

		dat_1d = np.reshape(dat, (1, -1))
		with open(dat_fname, 'a+') as f:
			np.savetxt(f, dat_1d, delimiter='\t', fmt='%.2e')


	def save_batches(self):
		batches_fname = os.path.join(self.dir, 'batches_info.json')
		#first_key = list(self.batch_saves.keys())[0]

		with open(batches_fname, 'w+') as f:
			json.dump(self.batch_saves, f, indent=4)




##################### Multi run


class AC_NN(nn.Module):

	def __init__(self, N_inputs, H, N_outputs, **kwargs):
		super(AC_NN, self).__init__()

		self.actor_lin1 = nn.Linear(N_inputs, H)
		self.mu = nn.Linear(H, N_outputs)
		self.sigma = nn.Linear(H, N_outputs)

		self.critic_lin1 = nn.Linear(N_inputs, H)
		self.v = nn.Linear(H, 1)

		self.sigma_min = kwargs.get('sigma_min', 10**-3)

		self.layers = [
						self.actor_lin1,
						self.mu,
						self.sigma,
						self.critic_lin1,
						self.v,
						]


	def forward(self, x):

		y = relu6(self.critic_lin1(x))
		v = self.v(y)

		z = relu6(self.actor_lin1(x))
		mu = self.mu(z)
		mu = 3*torch.tanh(mu)

		sigma = softplus(self.sigma(z)) + self.sigma_min
		return(v, (mu, sigma))



















#
