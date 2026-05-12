import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import functions as f
from scipy.sparse.linalg import eigsh

p = 0.25
N = 500
g = nx.erdos_renyi_graph(n=N, p=2.*8/(N-1.0))
sparse_g = nx.to_scipy_sparse_array(g)
eigenvalues, _ = eigsh(sparse_g.astype(float))
eig = np.min(eigenvalues) # lambda_N
sigma,_ = f.old_optimal_sigma(sparse_g, endVal = eig)
print(sigma*eig) # optimal sigma in terms of lambda_N
psi = f.domirank(g, sigma=sigma)
attack = f.generate_attack(psi)

lcc, links = f.network_attack_recovery(g, p, attack)
lcc2, links2 = f.network_attack_recovery(g, p, attack, method = "sequential")


plt.plot(np.linspace(0, 1, len(lcc2)-1), lcc2[:-1], label = f"sequential", linewidth=2)
plt.plot(np.linspace(0, 1, len(lcc)-1), lcc[:-1], label = f"random", linewidth=2)
plt.legend()
plt.ylim(0, 1.1)
plt.show()
