import numpy as np
import matplotlib.pyplot as plt

m = 1
b = 2
k = 3

A = np.array([[0,1],[-k/m,-b/m]])
B = np.array([0,1/m])
C = np.array([1,0])

T = 20
dt = 0.001

track = 2
if track == 1:
    kp,kd,ki=4,3,3
    actuator_model = 0
    r = lambda t:3

if track == 2:
    kp=1
    kd=2
    ki=15
    actuator_model = 0
    T = 100
    dt = .01
    r = lambda t:0.1*np.sin(.25*t)

if track == 3:
    kp,kd,ki = 3,2,.25
    r = lambda t:.25
    actuator_model = 1

x = np.array([0,0])
off_start_time = -1

all_x = x
t_prev =-100;e_prev=0;I=0
t_span = np.arange(0,T,dt)
u_prev = 0
u_stack = []
for t in t_span.tolist():
    e = r(t)-x[0]

    P = kp*e
    if t_prev>-1:
        I = I+ki*e*(t-t_prev)
        D = kd*(e-e_prev)/(t-t_prev)
        u = (P+I+D)
    else:
        u=P
    
    if(actuator_model):
        if(u>0):
            if u_prev == 0 and t-off_start_time<0.05:
                u=0
            else:
                u=1
                if u_prev == 0:
                    fire_start_time = t
        if(u<=0):
            if u_prev == 1 and t-fire_start_time<0.05:
                u = 1
            else:
                u = 0
                if u_prev == 1:
                    off_start_time = t

    e_prev = e
    t_prev = t
    u_prev = u

    x = x+dt*(A@x+B*u)
    all_x=np.vstack((all_x,x))
    u_stack.append(u)

fig, axs = plt.subplots(2)
axs[0].set_title('Response')
axs[0].plot(t_span,all_x[:][:-1])
output = r(t_span)
if np.isscalar(output):
    ref_plot = axs[0].plot(t_span,[r(t_span)]*len(t_span))
else:
    ref_plot = axs[0].plot(t_span,r(t_span))
axs[0].legend(["x(t)","x'(t)","r(t)"])
axs[1].set_title('Input')
axs[1].plot(t_span,u_stack)
plt.show()