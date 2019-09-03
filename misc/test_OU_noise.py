import numpy as np
import matplotlib.pyplot as plt

class OrnsteinUhlenbeckActionNoise():
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
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



OU_noise = OrnsteinUhlenbeckActionNoise(np.array([0]), sigma=0.1, theta=.01, dt=0.01)

fig, axes = plt.subplots(1, 2, figsize=(12,6))

N_pts = 2000
for n in range(3):
    noise = np.array([OU_noise() for i in range(N_pts)])
    path = np.cumsum(noise)
    axes[0].plot(noise)
    axes[1].plot(path)

#plt.savefig('OU_noise_ex.png')
plt.show()









#
