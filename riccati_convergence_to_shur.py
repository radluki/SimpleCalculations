import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are as scare

with open('riccati_hamiltonian.pickle','rb') as f:
    ham = pickle.load(f)
    t = ham[0]
    ham = ham[1]

with open('riccati.pickle','rb') as f:
    ric = pickle.load(f)
    ric = ric[1]


A = np.array([[-1,1],[0,-3]])
B = np.array([[1],[1]])
R = np.array([[0.5]])
Q = np.eye(2)
H1 = np.concatenate((A,-B.dot(np.linalg.inv(R)).dot(B.T)),axis=1)
H2 = np.concatenate((-Q,-A.T),axis=1)
H = np.concatenate((H1,H2),axis=0)
Qf = np.eye(2)
XT = np.eye(2)

P = scare(A,B,Q,R)
ham = [(np.log10(np.abs(k-P))).tolist() for k in ham]
ham = np.array(ham)
plt.subplot(211)
plt.plot(t,ham[:,1,0],'--',label=r'$p_{21}$')
plt.plot(t,ham[:,1,1],label=r'$p_{22}$')
plt.plot(t,ham[:,0,0],label=r'$p_{11}$')
plt.plot(t,ham[:,0,1],'-.',label=r'$p_{12}$')
plt.legend()
plt.title(r"$\log{(P_t^h-P^s)}$")

plt.subplot(212)
ric = [(np.log10(np.abs(k-P))).tolist() for k in ric]
ric = np.array(ric)
plt.plot(t,ric[:,1,0],'--',label=r'$p_{21}$')
plt.plot(t,ric[:,1,1],label=r'$p_{22}$')
plt.plot(t,ric[:,0,0],label=r'$p_{11}$')
plt.plot(t,ric[:,0,1],'-.',label=r'$p_{12}$')
plt.legend()
plt.title(r"$\log{(P_t^r-P^s)}$")

plt.show()