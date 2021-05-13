#%%
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
from control import lqr
import mplcursors

class smd_system:
    def __init__(self,m,b,k):
        self.m = m
        self.b = b
        self.k = k

        #x state is [x,xdot]
        self.A = np.array([[0,1],[-self.k/self.m,-self.b/self.m]])
        self.B = np.array([0,1/self.m])
        self.C = np.array([1,0])
        self.D = np.array([0])
        
        #sin p state is [p,pdot]
        w0 = 1
        self.Ac = np.array([[0,1],[-w0**2,0]])
        self.Bc = np.array([0,w0])

        # double integrator
        # self.Ac = np.array([[0,1],[0,0]])
        # self.Bc = np.array([0,1])
    
        self.Abar = np.block([[self.A,np.zeros((2,2))],
                             [-np.outer(self.Bc,self.C), self.Ac]])
        self.Bbar = np.concatenate([self.B,np.zeros(2)])
        self.Rbar = np.concatenate([np.zeros(2),self.Bc])

        self.K = np.array([10,10,-10,-10])

    def plant(self,t,p,e):
        return self.A@p+self.B*e
    
    def closedLoop(self,t,x,reference):
        return (self.Abar-np.outer(self.Bbar,self.K))@x+self.Rbar*reference(t)

    def simulate(self,x0,t0,tf,reference):
        sol = solve_ivp(self.closedLoop,[t0,tf],x0,args=([reference]), dense_output=True)
        t = np.linspace(t0, tf, 100)
        z = sol.sol(t)
        pos_plot = plt.plot(t,z[0])
        vel_plot = plt.plot(t,z[1])
        output = reference(t)
        if np.isscalar(output):
            ref_plot = plt.plot(t,[reference(t)]*len(t))
        else:
            ref_plot = plt.plot(t,reference(t))
        plt.xlabel('t')
        plt.ylabel('magnitude')
        plt.legend(["x'(t)", "x(t)", "r(t)"], shadow=True)
        plt.title('simulation')
        mplcursors.cursor(pos_plot+vel_plot)
        plt.show()

#%%
mysmd = smd_system(m=1,b=2,k=3)
# mysmd.simulate(np.array([5,5,0,0]),0,30,lambda x:10)
# %%
mysmd.simulate(np.array([5,5,0,0]),0,30,lambda x:np.sin(1*x))