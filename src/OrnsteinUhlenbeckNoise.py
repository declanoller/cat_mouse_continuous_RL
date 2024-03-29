import numpy as np


class OrnsteinUhlenbeckNoise():
	def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
		# mu is the "center" it creates noise around
		# x0 is where it starts
		# sigma is how much noise
		# theta is how fast it reaches mu, I think
		# dt is the time scale of correlation, I think, where smaller dt means longer
		# correlation?
		self.theta = theta
		self.mu = mu
		self.sigma = sigma
		self.dt = dt
		self.x0 = x0
		self.reset()

	def __call__(self):
		x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
		self.x_prev = x
		return x

	def reset(self):
		self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

	def __repr__(self):
		return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)
