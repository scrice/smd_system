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
    def __init__(self,m,b,k,mode,w=0):
        self.m = m
        self.b = b
        self.k = k

        #x state is [x,xdot]
        self.A = np.array([[0,1],[-self.k/self.m,-self.b/self.m]])
        self.B = np.array([[0,1/self.m]]).T
        self.C = np.array([[1,0]])
        
        if mode == 1:
            #single integrator
            self.Ac = np.array([[0]])
            self.Bc = np.array([[1]])
            self.Cc = np.array([[1]])
            self.K = np.array([[-41.6238,-7.8614,100.0000]]) #LQR magic

        if mode == 2:
            #sin p state is [p,pdot]
            w0 = 0.25 
            self.Ac = np.array([[0,1],[-w0**2,0]])
            self.Bc = np.array([[0,1]]).T
            self.Cc = np.array([0,1])
            self.K = np.array([[-1.1072,-0.4949,0.2479,4.0020]]) #LQR magic

        #Reference free closed loop system
        Abar = np.block([[self.A,np.zeros((np.shape(self.A)[0],np.shape(self.Ac)[1]))],[-self.Bc*self.C,self.Ac]])
        Bbar = np.block([[self.B],[np.zeros((np.shape(self.Ac)[0],1))]])
        #LQR magic would happen here

        #Closed loop system state space
        self.Acl = Abar+Bbar*self.K
        self.Bcl = np.block([[np.zeros(np.shape(self.B))],[self.Bc]])
        self.Ccl = np.block([self.C,np.zeros(np.shape(self.Cc))])

    def closedLoop(self,t,x,reference):
        return self.Acl@x+self.Bcl.flatten()*reference(t)
    
    def simulate(self,x0,t0,tf,reference):
        sol = solve_ivp(self.closedLoop,[t0,tf],x0,args=([reference]), dense_output=True)
        t = np.linspace(t0, tf, 100)
        z = sol.sol(t)
        y = self.Ccl@z
        sol_plot = plt.plot(t,y.flatten())
        output = reference(t)
        if np.isscalar(output):
            ref_plot = plt.plot(t,[reference(t)]*len(t))
        else:
            ref_plot = plt.plot(t,reference(t))
        plt.xlabel('time(s)')
        plt.ylabel('distance(m)')
        plt.legend(["x(t)", "r(t)"], shadow=True)
        plt.title('simulation')
        # mplcursors.cursor(sol_plot)
        plt.show()

#%%
const_ref_smd = smd_system(m=1,b=2,k=3,mode=1)
const_ref_smd.simulate(np.array([0,0,0]),0,5,lambda x:5)
# %%
wref = 0.25
sin_ref_smd = smd_system(m=1,b=2,k=3,mode=2,w=wref)
sin_ref_smd.simulate(np.array([0,0,0,0]),0,50,lambda x:0.1*np.sin(wref*x))