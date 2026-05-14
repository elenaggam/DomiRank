import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import functions as f
from scipy.sparse.linalg import eigsh

p = 0.25
N = 500
g = [nx.erdos_renyi_graph(n=N, p = 4./(N-1)), nx.barabasi_albert_graph(n=N, m=2)]
names = ['ER', 'BA']

for i in range(2):
    for method in ['random', 'sequential']:
        sparse = nx.to_scipy_sparse_array(g[i])
        eigenvalues, _ = eigsh(sparse.astype(float))
        eig = np.min(eigenvalues) # lambda_N
        sigma,_ = f.old_optimal_sigma(sparse, endVal = eig)

        # domirank
        psi = f.domirank(g[i], sigma=sigma)
        attack = f.generate_attack(psi)
        lcc, links = f.network_attack_recovery(g[i], p, attack, method = method)
        plt.plot(np.linspace(0, 1, len(lcc)-1), lcc[:-1], label = f"domirank (σ={sigma*eig:.2f}/λ)", linewidth=2,zorder=10)
        big_sigma = -0.999/eig
        if sigma < big_sigma:
            psi = f.domirank(g[i], sigma=big_sigma)
            attack = f.generate_attack(psi)
            lcc, links= f.network_attack_recovery(g[i], p, attack, method = method)
            plt.plot(np.linspace(0, 1, len(lcc)-1), lcc[:-1], label = f"domirank (σ={big_sigma*eig:.2f}/λ)", linewidth=2,  zorder=10)

        # betweenness
        attack = []
        lcc, links = f.network_attack_recovery(g[i], p, attack, centrality_func = nx.betweenness_centrality, method = method)
        plt.plot(np.linspace(0, 1, len(lcc)-1), lcc[:-1], label = f"betweenness", linewidth=2, color = 'black')

        # collective influence
        attack = []
        lcc, links = f.network_attack_recovery(g[i], p, attack, centrality_func = f.collective_influence, method = method)
        plt.plot(np.linspace(0, 1, len(lcc)-1), lcc[:-1], label = f"collective influence", linewidth=2, color = 'grey')

        plt.legend(loc = 'lower right')
        plt.ylim(0, 1.1)
        plt.title(f"{names[i]} - {method} recovery")
        plt.savefig(f"{names[i]}/recovery_{method}.png", bbox_inches='tight')
        plt.close()