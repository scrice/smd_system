import numpy as np
import matplotlib.pyplot as plt

m = 1
b = 2
k = 3

A = np.array([[0,1],[-k/m,-b/m]])
B = np.array([0,1/m])
C = np.array([1,0])

T = 50
dt = 0.1

track = 2
if track == 1:
    kp=1
    kd=2
    ki=3
    r = lambda t:3

if track == 2:
    kp=3
    kd=2
    ki=15
    r = lambda t:0.1*np.sin(.25*t)

x = np.array([2,0])

all_x = x
t_prev =-100;e_prev=0;I=0
t_span = np.arange(0,T,dt)
for t in t_span.tolist():
    e = r(t)-x[0]

    P = kp*e
    if t_prev>-1:
        I = I+ki*e*(t-t_prev)
        D = kd*(e-e_prev)/(t-t_prev)
        u = (P+I+D)
    else:
        u=P

    e_prev = e
    t_prev = t

    x = x+dt*(A@x+B*u)
    all_x=np.vstack((all_x,x))

plt.plot(t_span,all_x[:][:-1])
output = r(t_span)
if np.isscalar(output):
    ref_plot = plt.plot(t_span,[r(t_span)]*len(t_span))
else:
    ref_plot = plt.plot(t_span,r(t_span))
plt.legend(["x(t)","x'(t)","r(t)"])
plt.show()