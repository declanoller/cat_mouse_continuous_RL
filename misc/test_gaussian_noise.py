import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(8,6))

sigma = 0.1
for n in range(3):
    noise = sigma*np.random.randn(200)
    path = np.cumsum(noise)
    axes[0].plot(noise)
    axes[1].plot(path)

plt.savefig('gaussian_noise_ex.png')
plt.show()









#
