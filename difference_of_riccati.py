import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('riccati_hamiltonian.pickle','rb') as f:
    ham = pickle.load(f)
    t = ham[0]
    ham = ham[1]

with open('riccati.pickle','rb') as f:
    ric = pickle.load(f)
    ric = ric[1]

diff = np.log10(np.abs(ham - ric)/np.abs(ric))
plt.plot(t,diff[:,1,0],'--',label=r'$p_{21}$')
plt.plot(t,diff[:,1,1],label=r'$p_{22}$')
plt.plot(t,diff[:,0,0],label=r'$p_{11}$')
plt.plot(t,diff[:,0,1],'-.',label=r'$p_{12}$')
plt.legend()
plt.show()