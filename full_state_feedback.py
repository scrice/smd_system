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
        
        #sin p state is [p,pdot]
        # w0 = 0.25 
        # self.Ac = np.array([[0,1],[-w0**2,0]])
        # self.Bc = np.array([0,1])
        # self.Cc = np.array([0,1])

        # double integrator
        self.K = np.array([-1,-1])
        self.F = 4 

        # Arows,Acols = np.shape(self.A) 
        # Acrows,Accols = np.shape(self.Ac) 
        # self.Abar = np.block([[self.Ac,np.zeros((Acrows,Acols))], [np.outer(self.B,self.Cc), self.A]])
        # self.Bbar = np.concatenate([self.Bc,np.zeros(np.size(self.B))])
        # self.Cbar = np.concatenate([np.zeros(np.size(self.Cc)),self.C])

    def plant(self,t,x,u):
        return self.A@x+self.B*u
    
    def closedLoop(self,t,x,reference):
        return (self.A+np.outer(self.B,self.K))@x+self.B*self.F*reference(t)

    def simulate(self,x0,t0,tf,reference):
        sol = solve_ivp(self.closedLoop,[t0,tf],x0,args=([reference]), dense_output=True)
        t = np.linspace(t0, tf, 100)
        z = sol.sol(t)
        y = self.C@z
        sol_plot = plt.plot(t,y)
        # vel_plot = plt.plot(t,z[3])
        # pos_plot = plt.plot(t,z[2])
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
# mysmd.simulate(np.array([0,9]),0,50,lambda x:3)
# %%
mysmd.simulate(np.array([0,2]),0,50,lambda x:0.1*np.sin(0.25*x))

# print(linalg.eig(mysmd.Abar))