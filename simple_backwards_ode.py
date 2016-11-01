import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

"""
Simple presentation of solving odes backward in time
"""

A = np.array([[-1,1],[0,-1]])

def lin_ode(x, t):
    xdot = np.dot(A,x)
    return xdot.ravel()

def backwards_lin_ode(x, t):
    return -lin_ode(x, t)

y0 = [1,1]
t = np.linspace(0,3,1e2)
y = odeint(lin_ode,y0, t)
plt.plot(t,y[:,0])
plt.plot(t,y[:,1])


t = np.linspace(0,3,10)
y = odeint(backwards_lin_ode, y[-1, :], t)
y1 = list(reversed(y[:,0]))
y2 = list(reversed(y[:,1]))
plt.plot(t,y1,'*')
plt.plot(t,y2,'*')

plt.show()