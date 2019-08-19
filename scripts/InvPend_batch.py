from LSTM1 import LSTM1
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.nn.functional import softplus, relu6
import matplotlib.pyplot as plt
import decorators
import os
from math import exp, log, sqrt
import FileSystemTools as fst
import pickle
import json
from CatMouseAgent import CatMouseAgent
import RunTools as rt
from torch.utils.tensorboard import SummaryWriter
from GymAgent import GymAgent
import traceback as tb

all_kwargs = {
	'N_state_terms' : 0,
	'N_actions' : 0,
	'N_batch' : 5,
	'gamma' : 0.5,
	'beta_GAE' : 0.5,
	'beta_entropy' : 0.005,
	'beta_entropy_min' : 0.03,
	'beta_entropy_max' : 1.0,
	'entropy_method' : 'const',
	'beta_V_loss' : 0.25,
	'optim' : 'RMSprop',
	'LR' : 2*10**-4,
	'clip_grad' : False,
	'clip_amount' : 1.0,
	'clamp_grad' : 10000.0,
	'hidden_size' : 50,
	'N_eps' : 1000,
	'N_steps' : 500,
	'init_weights' : True,
	'base_dir' : 'misc_runs'

}


class InvPend_batch:


	def __init__(self, **kwargs):

		self.agent = GymAgent(env_name='Pendulum')

		self.N_state_terms = len(self.agent.getStateVec())
		self.N_actions = 1

		for k,v in kwargs.items():
			if k in all_kwargs.keys():
				all_kwargs[k] = v
			else:
				print(f'Passed parameter {k} not in all_kwargs dict! Check')
				assert k in all_kwargs.keys(), 'key error'

		#self.fname_base = fst.paramDictToFnameStr(kwargs) + '_' + fst.getDateString()
		self.fname_base = 'InvPend_' + fst.getDateString()
		self.dir = os.path.join(all_kwargs['base_dir'], self.fname_base)
		print(self.dir)
		os.mkdir(self.dir)
		print(f'\n\nMade dir {self.dir} for run...\n\n')

		self.save_params_json(kwargs, 'params_passed')
		self.save_params_json(all_kwargs, 'params_all')

		self.N_batch = all_kwargs['N_batch']
		self.gamma = all_kwargs['gamma']
		self.beta_GAE = all_kwargs['beta_GAE']
		self.entropy_method = all_kwargs['entropy_method']
		self.beta_entropy = all_kwargs['beta_entropy']
		self.beta_entropy_min = all_kwargs['beta_entropy_min']
		self.beta_entropy_max = all_kwargs['beta_entropy_max']
		self.beta_V_loss = all_kwargs['beta_V_loss']
		self.optim = all_kwargs['optim']
		self.LR = all_kwargs['LR']
		self.hidden_size = all_kwargs['hidden_size']
		self.init_weights = all_kwargs['init_weights']
		self.clamp_grad = all_kwargs['clamp_grad']
		self.clip_grad = all_kwargs['clip_grad']
		self.clip_amount = all_kwargs['clip_amount']

		self.setup_NN()
		#self.tb_writer = SummaryWriter()
		self.dat_fname_file = None


	def save_params_json(self, d, fname):

		fname_abs = os.path.join(self.dir, fname + '.json')

		with open(fname_abs, 'w+') as f:
			json.dump(d, f, indent=4)


	def setup_NN(self):

		self.NN = AC_NN(self.N_state_terms, 200, 2)

		if self.init_weights:
			for l in self.NN.layers:
				nn.init.normal_(l.weight, mean=0., std=0.1)
				nn.init.constant_(l.bias, 0.)

		if self.optim == 'Adam':
			self.optimizer = optim.Adam(self.NN.parameters(), lr=self.LR)
		else:
			self.optimizer = optim.RMSprop(self.NN.parameters(), lr=self.LR)


	@decorators.timer
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

		self.R_train_hist = []

		self.batch_saves = {}
		self.global_step = 0

		R_avg = 0
		try:
			for ep in range(N_eps):
				self.episode(N_steps)

				R_tot = sum(self.last_ep_R_hist)
				if ep==0:
					R_avg = R_tot
				else:
					R_avg = 0.99*R_avg + 0.01*R_tot

				if ep % max(N_eps // 1000, 1) == 0:
					#self.plot_episode(show_plot=False, save_plot=True, iter=ep)
					#self.plot_VF(show_plot=False, save_plot=True, iter=ep)
					print('\nepisode {}/{}'.format(ep, N_eps))
					print(f'Episode R_tot = {R_tot:.4f}\t avg reward = {R_avg:.1f}')


		except:
			print('\n\nRun stopped for some reason.\n\n')
			print(tb.format_exc())
			print('\n\n')

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
		a_ep_hist = []
		R_ep_hist = []

		# Needed for calculating loss
		v_buffer = []
		r_buffer = []
		sigma_buffer = []
		pi_a_buffer = []

		# Others
		s_buffer = []
		a_buffer = []
		mu_buffer = []


		for t in range(N_steps):

			s = self.agent.getStateVec()

			out_V, out_pi = self.NN.forward(torch.tensor(s, dtype=torch.float))
			mu, sigma = out_pi
			mu = mu.squeeze()
			sigma = sigma.squeeze()
			# Calculate the dist. we're gonna draw from.
			a = np.random.normal(loc=mu.detach(), scale=max(0, sigma.detach()))

			pi_val = (1.0/(sigma*sqrt(2*3.14)))*torch.exp(-(a - mu).pow(2)/(sigma.pow(2)*2))

			r, s_next, done = self.agent.iterate(a)
			R_ep_hist.append(r)
			r = (r + 8.0)/8.0 # Normalize r

			# Needed to calc loss
			sigma_buffer.append(sigma)
			v_buffer.append(out_V)
			pi_a_buffer.append(pi_val)
			r_buffer.append(r)

			# Others, debugging
			s_buffer.append(s.tolist())
			a_buffer.append(a)
			mu_buffer.append(mu.detach().item())

			# Ep hist stuff
			s_ep_hist.append(s)
			a_ep_hist.append(a)
			sigma_ep_hist.append(sigma)

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
				policy_loss = (-adv.detach()*torch.log(pi_batch))
				V_loss = adv.pow(2)
				H_loss = -self.beta_entropy*(0.5 + 0.5*log(2 * 3.14) + torch.log(sigma_batch))

				loss_tot = (policy_loss + V_loss + H_loss).mean()

				self.optimizer.zero_grad()
				loss_tot.backward()

				grad_norm_thresh = 500.0

				total_norm = 0
				for p in self.NN.parameters():
					param_norm = p.grad.data.norm(2.0)
					total_norm += param_norm.item() ** 2.0

				total_norm = total_norm ** (1. / 2.0)
				self.save_dat_append('grad_norm', np.array([total_norm]))

				if total_norm > grad_norm_thresh:
					print(f'Grad norm ({total_norm:.2f} above threshold ({grad_norm_thresh}))')
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

					}

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
		self.last_ep_SD_hist = np.array(sigma_ep_hist)
		self.last_ep_action_hist = np.array(a_ep_hist)




	def smooth_data(self, in_dat):

		hist = np.array(in_dat)
		N_avg_pts = min(100, len(hist))

		avg_period = max(1, len(hist) // max(1, N_avg_pts))

		downsampled_x = avg_period*np.array(range(N_avg_pts))
		hist_downsampled_mean = [hist[i*avg_period:(i+1)*avg_period].mean() for i in range(N_avg_pts)]
		return(downsampled_x, hist_downsampled_mean)



	def plot_VF(self, **kwargs):

		plt.close('all')

		plt.figure(figsize=(8,8))
		ax = plt.gca()

		z = 30
		N_grid_theta = z
		N_grid_thetadot = z

		theta = np.linspace(-3.14, 3.14, N_grid_theta)

		cos_theta = np.cos(theta)
		sin_theta = np.sin(theta)
		theta_dot = np.linspace(-8, 8, N_grid_thetadot)

		# The coords go like a matrix, so the first one is the y direction
		V = np.zeros((N_grid_theta, N_grid_thetadot))

		for i,x in enumerate(zip(cos_theta, sin_theta)):
			for j,y in enumerate(theta_dot):

				s = torch.tensor([*x, y], dtype=torch.float)
				out_v, out_pi = self.NN.forward(s)
				V[i, j] = out_v.item()

		cax = ax.matshow(V)
		cbar = plt.colorbar(cax, fraction=0.046, pad=0.04)
		plt.xlabel('theta dot', fontsize=16)
		plt.ylabel('theta', fontsize=16)
		ax.set_xticks([0, N_grid_thetadot-1])
		ax.set_xticklabels(['-8', '8'])
		ax.set_yticks([0, N_grid_theta-1])
		ax.set_yticklabels(['-pi', 'pi'])

		if kwargs.get('save_plot', True):
			if kwargs.get('iter', None) is not None:
				plt.savefig(os.path.join(self.dir, '{}_VF_{}.png'.format(self.fname_base, kwargs.get('iter'))))
			else:
				plt.savefig(os.path.join(self.dir, '{}_VF.png'.format(self.fname_base)))
		if kwargs.get('show_plot', True):
			plt.show()

		plt.close('all')



	def plot_episode(self, **kwargs):

		plt.close('all')

		fig, axes = plt.subplots(2, 4, figsize=(14,8))

		ax_cos = axes[0][0]
		ax_sin = axes[0][1]
		ax_theta_dot = axes[0][2]
		ax_R = axes[0][3]

		ax_actions = axes[1][0]
		ax_SD = axes[1][1]
		ax_noise = axes[1][2]

		ax_cos.set_xlabel('step')
		ax_cos.set_ylabel('cos(ang)')
		ax_cos.plot(self.last_ep_state_hist[:,0], color='dodgerblue')
		ax_cos.set_ylim(-1, 1)

		ax_sin.set_xlabel('step')
		ax_sin.set_ylabel('sin(ang)')
		ax_sin.plot(self.last_ep_state_hist[:,1], color='tomato')
		ax_sin.set_ylim(-1, 1)

		ax_theta_dot.set_xlabel('step')
		ax_theta_dot.set_ylabel('ang_vel')
		ax_theta_dot.plot(self.last_ep_state_hist[:,2], color='forestgreen')
		ax_theta_dot.set_ylim(-8, 8)

		ax_R.set_xlabel('step')
		ax_R.set_ylabel('R')
		ax_R.plot(self.last_ep_R_hist, color='darkorange')
		ax_R.set_ylim(-16, 0)


		ax_actions.set_xlabel('step')
		ax_actions.set_ylabel('action')
		ax_actions.plot(self.last_ep_action_hist, color='turquoise')

		ax_SD.set_xlabel('step')
		ax_SD.set_ylabel('SD')
		ax_SD.plot(self.last_ep_SD_hist, color='darkorange')

		plt.tight_layout()

		if kwargs.get('save_plot', True):
			if kwargs.get('iter', None) is not None:
				plt.savefig(os.path.join(self.dir, '{}_traj_{}.png'.format(self.fname_base, kwargs.get('iter'))))
			else:
				plt.savefig(os.path.join(self.dir, '{}_traj.png'.format(self.fname_base)))
		if kwargs.get('show_plot', True):
			plt.show()



	def plot_train(self, **kwargs):

		plt.close('all')

		fig, axes = plt.subplots(2, 3, figsize=(12,8))

		ax_R = axes[0][0]
		ax_R_episode = axes[0][2]
		ax_H_loss = axes[0][1]
		ax_V_loss = axes[1][0]
		ax_policy_loss = axes[1][1]


		ax_R.set_xlabel('ep')
		ax_R.set_ylabel('R')
		ax_R.plot(self.R_batches, color='dodgerblue')

		ax_R_episode.set_xlabel('ep')
		ax_R_episode.set_ylabel('R')
		ax_R_episode.plot(self.R_train_hist, color='dodgerblue', alpha=0.5)
		ax_R_episode.plot(*self.smooth_data(self.R_train_hist), color='black')
		ax_R_episode.set_ylim(-1700, 50)

		ax_V_loss.set_xlabel('ep')
		ax_V_loss.set_ylabel('V_loss')
		ax_V_loss.plot(self.V_loss_batches, color='tomato')

		ax_H_loss.set_xlabel('ep')
		ax_H_loss.set_ylabel('H_loss')
		ax_H_loss.plot(self.H_loss_batches, color='forestgreen')

		ax_policy_loss.set_xlabel('ep')
		ax_policy_loss.set_ylabel('policy_loss')
		ax_policy_loss.plot(self.pol_loss_batches, color='darkorange')

		plt.tight_layout()

		if kwargs.get('save_plot', True):
			if kwargs.get('iter', None) is not None:
				plt.savefig(os.path.join(self.dir, '{}_train_{}.png'.format(self.fname_base, kwargs.get('iter'))))
			else:
				plt.savefig(os.path.join(self.dir, '{}_train.png'.format(self.fname_base)))
		if kwargs.get('show_plot', True):
			plt.show()



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

def search_param(**kwargs):

	cm = InvPend_batch(**kwargs)

	'''base_dir = kwargs.get('base_dir', 'misc_runs')
	run_dir = os.path.join(base_dir, cm.fname_base)
	os.mkdir(run_dir)
	cm.dir = run_dir'''

	#cm.plot_trajectory(show_plot=False)
	#cm.plot_episode(show_plot=False)
	cm.train(kwargs.get('N_eps', 200), kwargs.get('N_steps', 500))

	return(cm.smooth_data(cm.R_train_hist))


class AC_NN(nn.Module):

	def __init__(self, N_inputs, H, N_outputs):
		super(AC_NN, self).__init__()

		self.actor_lin1 = nn.Linear(N_inputs, 200)
		self.mu = nn.Linear(200, 1)
		self.sigma = nn.Linear(200, 1)

		self.critic_lin1 = nn.Linear(N_inputs, 100)
		self.v = nn.Linear(100, 1)

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
		mu = 2*torch.tanh(self.mu(z))
		sigma = softplus(self.sigma(z)) + 0.001
		#mu, g = self.actor_lin2(z)
		#sigma = softplus(g) + 0.001 # makes it so the dist. can't get too narrow
		#print(v, (mu, sigma))
		return(v, (mu, sigma))





if __name__ == '__main__':





	base_dir = os.path.join(os.path.join(os.path.dirname(__file__), 'misc_runs'), 'vary_param_{}'.format(fst.getDateString()))
	os.mkdir(base_dir)

	center_dict = {
		'clip_grad' : False,
		'N_eps' : 200,
		'N_steps' : 500,
		'gamma' : 0.9,
		'LR' : 10**-3,
		'N_batch' : 5,
		'base_dir' : base_dir
	}

	vary_param_dict = {
		'LR' : [10**-2, 10**-4],
		'N_batch' : [2, 10]

	}

	rt.hyperparam_search_const(search_param,
								center_dict,
								vary_param_dict,
								N_runs=2,
								run_dir=base_dir,
								save_R=True)

	exit()



	cm = InvPend_batch(
				init_weights=True,
				gamma=0.9,
				LR=2*10**-4,
				beta_entropy=0.005,
				optim='Adam',
				clamp_grad=10000
				)


	cm.train(2000, 500)
	cm.plot_train(save_plot=True, show_plot=False)
	#cm.save_batches()
	#cm.plot_trajectory()
	#cm.plot_episode()

	exit()















#
