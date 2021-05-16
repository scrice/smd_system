import control as ctrl
import numpy as np
from control.timeresp import forced_response, impulse_response, step_response
from control.xferfcn import TransferFunction
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
from scipy import linalg

m = 1
b = 2
k = 3

const_ref = 5
w = 0.25
sin_ref = lambda x:0.1*np.sin(w*x)

kp = 2;kd =2;ki = 5;ks = 5

G = ctrl.TransferFunction(1,[m,b,k])
#Transfer function is (kds^2+kps+ki)/s
C1 = ctrl.TransferFunction([kd,kp,ki],[1,0])
C2 = ctrl.TransferFunction([kp,ki+ks,kp*w**2,ki*w**2+kp*w**2],[1,0,w**2])

C = C2
open_loop = G*C
closed_loop = ctrl.feedback(G*C,1)

t1,y1 = impulse_response(TransferFunction(const_ref,[1,0])*closed_loop,T=50)

sintf = TransferFunction(0.1*w,[1,0,w**2])
t2,y2 = impulse_response(sintf*closed_loop,T=50)

fig, axs = plt.subplots(2)
fig.suptitle('Two Responses')
axs[0].plot(t1, y1)
# axs[0].plot(t1,const_ref*ones(len(t1)))
axs[1].plot(t2, y2)
axs[1].plot(t2, sin_ref(t2))
axs[1].legend(['x','r'])
plt.show()

print(ctrl.pole(open_loop))
print(ctrl.pole(closed_loop))
# ctrl.lqr(open_loop,np.array([[1,0],[0,1]]),1)

hold =1 