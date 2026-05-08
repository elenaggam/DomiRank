import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import functions as f
from scipy.sparse.linalg import eigsh

N = 100

g = nx.watts_strogatz_graph(n=N, k=4, p=0.12)
sparse_g = nx.to_scipy_sparse_array(g)
eigenvalues, _ = eigsh(sparse_g.astype(float))
eig = np.min(eigenvalues) # lambda_N

sigma, areas_lcc = f.old_optimal_sigma(sparse_g, endVal = eig, iterationNo = 300, sampling = 1, directory = 'critico/')

psi = f.domirank(g, sigma=sigma)
attack = f.generate_attack(psi)
lcc, links = f.network_attack_sampled(g, attack, sampling = 1)
plt.plot(np.linspace(0, 1, len(lcc)), lcc, label = f"{sigma*eig:.2f}", color = "b", linestyle='dotted', linewidth=3, zorder=10)

sigma = -0.89/eig
psi = f.domirank(g, sigma=sigma)
attack = f.generate_attack(psi)
lcc, links = f.network_attack_sampled(g, attack, sampling = 1)
plt.plot(np.linspace(0, 1, len(lcc)), lcc, label = f"{sigma*eig:.2f}", color = "r", linestyle='dotted', linewidth=3, zorder=10)
plt.legend()
plt.savefig("critico/sigma_optima_2x.png", dpi=300, bbox_inches='tight')


