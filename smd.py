#%%
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
from mpldatacursor import datacursor

class smd_system:
    def __init__(self,m,b,k):
        self.m = m
        self.b = b
        self.k = k
        self.A = np.array([[0,1,0],[-self.k/self.m,-self.b/self.m,0],[0,-1,0]])
        self.B = np.array([0,1/self.m,1])
        self.C = np.array([1,0])
        self.D = np.array([0])

    def derivative(self,t,x,controller,reference):
        Kp = 3
        Kd = 2
        Ki = 1
        #u = -Kp*(x[0]-reference)-Kd*(x[1])-Ki*x[2]
        u = -Kp*(x[0]-reference)-Kd*(x[1])-Ki*x[2]
        return self.A@x+self.B*u

    def simulate(self,x0,t0,tf,reference):
        sol = solve_ivp(self.derivative,[t0,tf],x0, args=(1,reference),dense_output=True)
        t = np.linspace(t0, tf, 100)
        z = sol.sol(t)
        plt.plot(t,z[0])
        plt.plot(t,z[1])
        plt.xlabel('t')
        plt.legend(["x(t)", "x'(t)"], shadow=True)
        plt.title('simulation')
        datacursor()
        plt.show()

#%%
mysmd = smd_system(1,2,3)
mysmd.simulate(np.array([5,5,0]),0,30,3)
# %%
