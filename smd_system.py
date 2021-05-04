import numpy as np
from scipy.integrate import RK45
from scipy.integrate import solve_ivp

class SMD:
    def __init__(self,m,b,k):
        self.m = m
        self.b = b
        self.k = k

    def linear_dynamics(self,t,x):
        A = np.array([[0,1],[-self.k/self.m,-self.b/self.m]])
        B = np.array([0,1/self.m])
        C = np.array([1,0])
        D = np.array([0])

        dxdt = A@x+B*1
        return dxdt

    def simulate(self):
       solve_ivp(my_SMD.linear_dynamics, [0,10], x0)

m = 1
b = 2
k = 3
x0 = np.array([0,0])

my_SMD = SMD(m,b,k)
my_SMD.simulate()
output  = solve_ivp(my_SMD.linear_dynamics, [0,5], x0)

print(output.y)