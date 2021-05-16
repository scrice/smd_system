#For thruster problem see Sidi(1997) and Wie(2008)

#%%
import numpy as np
from numpy.core.fromnumeric import size
from scipy import linalg
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import mplcursors

class smd_system:
    def __init__(self,m,b,k):
        self.m = m
        self.b = b
        self.k = k

        #x state is [x,xdot]
        self.A = np.array([[-self.b/self.m,-self.k/self.m],[1,0]])
        self.B = np.array([1/self.m,0])
        self.C = np.array([0,1])
        
        self.K = np.array([-2,-1])
        self.F = 4 

    def plant(self,t,x,u):
        return self.A@x+self.B*u
    
    def closedLoop(self,t,x,reference):
        u = self.K@x+self.F*reference(t)
        return self.A@x+self.B*u

    def actuatorModel(self,t,x,reference):
        u = self.K@x+self.F*reference(t)
        return self.A@x+self.B*u

    def simulate(self,x0,t0,tf,reference):
        sol = solve_ivp(self.closedLoop,[t0,tf],x0,args=([reference]), dense_output=True)
        t = np.linspace(t0, tf, 100)
        z = sol.sol(t)
        y = self.C@z
        sol_plot = plt.plot(t,y)
        output = reference(t)
        if np.isscalar(output):
            ref_plot = plt.plot(t,[reference(t)]*len(t))
        else:
            ref_plot = plt.plot(t,reference(t))
        plt.xlabel('t')
        plt.ylabel('magnitude')
        plt.legend(["x(t)", "r(t)"], shadow=True)
        plt.title('simulation')
        # mplcursors.cursor(sol_plot)
        plt.show()

#%%
mysmd = smd_system(m=1,b=2,k=3)
mysmd.simulate(np.array([0,0]),0,10,lambda x:.25)
# %%
# mysmd.simulate(np.array([0,2]),0,50,lambda x:0.1*np.sin(0.25*x))