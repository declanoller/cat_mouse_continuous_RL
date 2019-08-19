
from random import randint,random,sample
import numpy as np
from math import atan,sin,cos,sqrt,ceil,floor,log
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import FileSystemTools as fst
from collections import namedtuple

from LSTM1 import LSTM1



class Agent:

	def __init__(self, **kwargs):

		self.agent_class = kwargs.get('agent_class',None)
		self.agent_class_name = self.agent_class.__name__
		self.agent = self.agent_class(**kwargs)

		self.params = self.agent.getPassedParams()

		#Eventually, it would be good to make it so it only actually adds to the filename
		#the parameters that were specifically passed to it, and adds the default ones
		#to the log file.
		self.params['Agent'] = self.agent_class_name

		self.params['epsilon'] = kwargs.get('epsilon',0.9)
		self.params['eps_decay'] = kwargs.get('eps_decay', 0.999)
		self.params['eps_min'] = kwargs.get('eps_min', 0.02)

		self.params['N_steps'] = kwargs.get('N_steps',10**3)
		self.params['N_batch'] = int(kwargs.get('N_batch',60))
		self.params['N_hidden_layer_nodes'] = int(kwargs.get('N_hidden_layer_nodes',100))
		self.params['target_update'] = int(kwargs.get('target_update',500))
		self.params['double_DQN'] = kwargs.get('double_DQN',False)

		self.params['features'] = kwargs.get('features','DQN')
		self.params['NL_fn'] = kwargs.get('NL_fn','tanh')
		self.params['loss_method'] = kwargs.get('loss_method','L2')

		self.params['advantage'] = kwargs.get('advantage', True)
		self.params['beta'] = kwargs.get('beta',0.1)

		self.dir = kwargs.get('dir','misc_runs')
		self.date_time = kwargs.get('date_time',fst.getDateString())
		self.base_fname = fst.paramDictToFnameStr(self.params) + '_' + self.date_time
		self.img_fname = fst.combineDirAndFile(self.dir, self.base_fname + '.png')
		self.log_fname = fst.combineDirAndFile(self.dir, 'log_' + self.base_fname + '.txt')
		self.reward_fname = fst.combineDirAndFile(self.dir, 'reward_' + self.base_fname + '.txt')
		self.weights_hist_fname = fst.combineDirAndFile(self.dir, 'weights_hist_' + self.base_fname + '.txt')
		self.model_fname = fst.combineDirAndFile(self.dir, 'model_' + self.base_fname + '.model')

		##### These are params we don't need in the basename.

		self.params['alpha'] = kwargs.get('alpha',10**-1)
		self.params['ACER'] = kwargs.get('ACER', False)

		self.params['exp_buf_len'] = int(kwargs.get('exp_buf_len',10000))
		self.params['gamma'] = kwargs.get('gamma',1.0)
		self.params['clamp_grad'] = kwargs.get('clamp_grad',False)

		self.figsize = kwargs.get('figsize', (12, 8))
		self.R_tot_hist = []


		self.setupLearn()





	def setupLearn(self):

		self.R_tot = 0
		self.scheduler = None

		self.N_actions = self.agent.N_actions #Should change this to a getter fn at some point.

		self.dtype = torch.float32
		torch.set_default_dtype(self.dtype)


		NL_fn_dict = {'relu':F.relu, 'tanh':torch.tanh, 'sigmoid':F.sigmoid}

		self.weights_history = []

		if self.params['features'] == 'AC':
			D_in, H, D_out = self.agent.N_state_terms, self.params['N_hidden_layer_nodes'], self.agent.N_actions

			NL_fn = NL_fn_dict[self.params['NL_fn']]

			#The "actor" is the policy network, the "critic" is the Q network.
			self.actor_NN = DQN(D_in, H, D_out, NL_fn=NL_fn, softmax=True)

			if self.params['advantage']:
				self.critic_NN = DQN(D_in, H, 1, NL_fn=NL_fn)
			else:
				self.critic_NN = DQN(D_in, H, D_out, NL_fn=NL_fn)

			self.actor_optimizer = optim.RMSprop(self.actor_NN.parameters())
			self.critic_optimizer = optim.RMSprop(self.critic_NN.parameters())
			self.experiences = []


	######### Functions for updating values periodically.


	def updateFrozenQ(self, iteration):

		if iteration%self.params['target_update']==0 and iteration>self.params['N_batch']:
			if self.params['features'] == 'AC':
				self.target_critic_NN.load_state_dict(self.critic_NN.state_dict())


	def updateR(self, r, iteration):
		self.R_tot += r.item()
		self.R_tot_hist.append(self.R_tot/(iteration+1))

		if iteration%int(self.params['N_steps']/10) == 0:
			print('iteration {}, R_tot/i = {:.3f}'.format(iteration, self.R_tot_hist[-1]))
			if self.scheduler is not None:
				print('LR: {:.4f}'.format(self.optimizer.param_groups[0]['lr']))



	########### Functions for interacting with networks.


	def softmaxAction(self, NN, state_vec):
		pi_vals = self.forwardPass(self.actor_NN, state_vec).detach()
		m = Categorical(pi_vals)
		return(m.sample().item())


	def forwardPass(self, NN, state_vec):
		#This will automatically figure out if it's a single tensor or a batch,
		#and if it's a single, unsqueeze it to the right dimension and then squeeze it
		#again after. It assumes that the network only has a single input dimension, though
		#(so it won't handle it if the input shape for the network is 3x5 or something).
		assert len(state_vec.shape)>=1, 'len of state vec is <1, error.'

		if len(state_vec.shape)==1:
			output = NN(state_vec.unsqueeze(dim=0)).squeeze()
		else:
			output = NN(state_vec)

		return(output)


	def getRandomAction(self):
		return(randint(0, self.N_actions-1))


	########### Functions for different episode types.



	def ACepisode(self):

		self.initEpisode()

		s = self.getStateVec()
		a = self.softmaxAction(self.actor_NN, s)

		for i in range(self.params['N_steps']):

			r, s_next = self.iterate(a)
			a_next = self.softmaxAction(self.actor_NN, s_next)


			if self.params['advantage']:
				critic_cur = self.forwardPass(self.critic_NN, s)
				critic_next = self.forwardPass(self.critic_NN, s_next)
				critic_value = (r + self.params['gamma']*critic_next.detach() - critic_cur.detach())
			else:
				critic_cur = self.forwardPass(self.critic_NN, s)[a]
				critic_next = self.forwardPass(self.critic_NN, s_next)[a_next]
				critic_value = critic_cur.detach()

			if self.params['loss_method'] == 'L2':
				TD0_error = (r + self.params['gamma']*critic_next - critic_cur).pow(2).sum()


			pi = self.forwardPass(self.actor_NN, s)
			iota = 10**-6
			pi = (pi + iota)/sum(pi + iota)
			#This sum is HIGHER for more entropy. So we want to put it in J with a negative, to maximize it. Err...seems backwards.
			entropy = -self.params['beta']*sum(pi*torch.log(pi))
			J = -(critic_value*torch.log(pi[a]) + entropy)


			self.actor_optimizer.zero_grad()
			self.critic_optimizer.zero_grad()

			J.backward(retain_graph=True)
			TD0_error.backward()

			self.actor_optimizer.step()
			self.critic_optimizer.step()

			s = s_next
			a = a_next

		#self.saveRewardCurve()

		print('self.R_tot/N_steps: {:.2f}'.format(self.R_tot/self.params['N_steps']))
		return(self.R_tot/self.params['N_steps'])



	########### These are functions for interacting with the agent object class.


	def iterate(self, a):
		r, s_next = self.agent.iterate(a)
		return(torch.tensor(r, dtype=self.dtype), torch.tensor(s_next, dtype=self.dtype))


	def initEpisode(self):
		self.agent.initEpisode()


	def resetStateValues(self):
		self.agent.resetStateValues()


	def getStateVec(self):
		return(torch.tensor(self.agent.getStateVec(), dtype=self.dtype))


	def getReward(self):
		return(torch.tensor(self.agent.reward(), dtype=self.dtype))




#
