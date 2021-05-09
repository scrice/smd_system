import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

x = np.array([0,2,5,7,10])
y = np.array([np.pi/2, 1, 2, -np.pi/4,  0])
cs = CubicSpline(x, y,bc_type='natural')
xs = np.arange(0, 10, 0.1)
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(x, y, 'o', label='data')
ax.plot(xs, cs(xs), label=r"$\theta(t)$")
ax.plot(xs, cs(xs, 1), label=r"$\theta'(t)$")
ax.plot(xs, cs(xs, 2), label=r"$\theta''(t)$")
ax.set_xlim(0, 10)
ax.legend(loc='lower left', ncol=2)
plt.title("Sample Single Joint Angle Over Time")
plt.xlabel("Time (s)")
plt.xlabel("Time (s)")
plt.ylabel("Joint angle (rad)")
plt.show()