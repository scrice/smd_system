#%%
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class smd_system:
    def __init__(self,m,b,k):
        self.m = m
        self.b = b
        self.k = k
        self.A = np.array([[0,1],[-self.k/self.m,-self.b/self.m]])
        self.B = np.array([0,1/self.m])
        self.C = np.array([1,0])
        self.D = np.array([0])

        #update this
        self.F = 1

    def derivative(self,t,x):
        return self.A@x+self.B*np.sin(t)

    def simulate(self,x0,t0,tf):
        sol = solve_ivp(self.derivative,[t0,tf],x0,dense_output=True)
        t = np.linspace(t0, tf, 300)
        z = sol.sol(t)
        plt.plot(t,z[0])
        plt.plot(t,z[1])
        plt.xlabel('t')
        plt.legend(["x(t)", "x'(t)"], shadow=True)
        plt.title('simulation')
        plt.show()



#%%
mysmd = smd_system(1,2,3)
mysmd.simulate(np.array([5,5]),0,10)
# %%
