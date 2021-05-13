#%%
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
from control import lqr
import mplcursors

#x state is [x,xdot]
A = np.array([[-2,-4.031,-1.531,-0.25,-0.125],
              [2,0,0,0,0],
              [0,2,0,0,0],
              [0,0,0.5,0,0],
              [0,0,0,0.25,0]]).T
C = np.array([2,0,0,0,0])
B = np.array([0,1.25,0.75,0.07812,0.0625])
D = 0

#sin p state is [p,pdot]
w0 = .25
# Ac = np.array([[0,1],[-w0**2,0]])
# Bc = np.array([0,w0])

# double integrator
# Ac = np.array([[0,1],[0,0]])
# Bc = np.array([0,1])

# Abar = np.block([[A,np.zeros((2,2))],
#                         [-np.outer(Bc,C), Ac]])
# Bbar = np.concatenate([B,np.zeros(2)])
# Rbar = np.concatenate([np.zeros(2),Bc])

# K = np.array([-463,-221,-2501,367])

def plant(t,x,reference):
    return A@x+B*reference(t)

def simulate(x0,t0,tf,reference):
    sol = solve_ivp(plant,[t0,tf],x0,args=([reference]), dense_output=True)
    t = np.linspace(t0, tf, 100)
    z = sol.sol(t)
    y = C@z
    pos_plot = plt.plot(t,y)
    output = reference(t)
    if np.isscalar(output):
        ref_plot = plt.plot(t,[reference(t)]*len(t))
    else:
        ref_plot = plt.plot(t,reference(t))
    plt.xlabel('t')
    plt.ylabel('magnitude')
    plt.legend(["x(t)", "r(t)"], shadow=True)
    plt.title('simulation')
    plt.show()

# mysmd.simulate(np.array([5,5,0,0]),0,30,lambda x:10)
# %%
simulate(np.array([1,1,1,1,1]),0,300,lambda x:0.1*np.sin(w0*x))

hold = 1