import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are
import pickle
"""
Solves Riccati Differential Equation backwards in time
"""
A = np.array([[-1,1],[0,-3]])
B = np.array([[1],[1]])
R = np.array([[0.5]])
Q = np.eye(2)
H1 = np.concatenate((A,-B.dot(np.linalg.inv(R)).dot(B.T)),axis=1)
H2 = np.concatenate((-Q,-A.T),axis=1)
H = np.concatenate((H1,H2),axis=0)
Qf = np.eye(2)
XT = np.eye(2)

n_size = A.shape[0]


def riccati(x, t):
    Pt = np.reshape(x,(n_size,n_size))
    X1 = np.dot(B,np.linalg.inv(R))
    X2 = np.dot(B.T,Pt)
    X1 = np.dot(X1,X2)
    mPtp =  np.dot(A.T,Pt) + np.dot(Pt,A) - np.dot(Pt,X1) + Q
    return mPtp.ravel()

t = np.linspace(0,30,1e3)
y0 = Qf.ravel()
y = odeint(riccati,y0, t)
y = list(reversed([ np.reshape(k,(n_size,n_size)).tolist() for k in y]))
pt = np.array(y)
plt.plot(t,pt[:,1,0],'--',label=r'$p_{21}$')
plt.plot(t,pt[:,1,1],label=r'$p_{22}$')
plt.plot(t,pt[:,0,0],label=r'$p_{11}$')
plt.plot(t,pt[:,0,1],'-.',label=r'$p_{12}$')
#plt.plot(det)
Pt = pt[0]
X = solve_continuous_are(A, B, Q, R)
print('Pt = ',Pt)
print('X =',X)
print('X-Pt =',X-Pt)
plt.legend()
with open('riccati.pickle','wb') as f:
    pickle.dump((t,pt),f)
plt.show()

