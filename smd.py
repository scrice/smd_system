#%%
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import mplcursors

class smd_system:
    def __init__(self,m,b,k):
        self.m = m
        self.b = b
        self.k = k

    def derivative(self,t,x,controller,reference):
        Kp = 3 
        Kd = 2
        Ki = 1
        u = Kp*(reference-x[1])+Kd*(-x[0])+Ki*(x[2])
        #return self.A@x+self.B*u
        return np.array([-self.k/self.m*x[1]-self.b/self.m*x[0]+1/self.m*u, x[0], reference-x[1]])

    def simulate(self,x0,t0,tf,reference):
        sol = solve_ivp(self.derivative,[t0,tf],x0, args=(1,reference),dense_output=True)
        t = np.linspace(t0, tf, 100)
        z = sol.sol(t)
        pos_plot = plt.plot(t,z[0])
        vel_plot = plt.plot(t,z[1])
        plt.xlabel('t')
        plt.legend(["x(t)", "x'(t)"], shadow=True)
        plt.title('simulation')
        mplcursors.cursor(pos_plot+vel_plot)
        plt.show()

#%%
mysmd = smd_system(1,2,3)
mysmd.simulate(np.array([5,5,0]),0,30,5)
# %%
