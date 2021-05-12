import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

sol = solve_ivp(lambda t,x:np.sin(t),[0,10],[-1],dense_output=True)
t = np.linspace(0, 10, 100)
z = sol.sol(t)
test_plot = plt.plot(t,z[0])
plt.show()