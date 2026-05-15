import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import functions as f
from scipy.sparse.linalg import eigsh
import functions_kuramoto as fk
import functions as f


net_name = 'ER'

N = 100
avg_k = 8
p2 = avg_k/(N-1)

steps = 1000
K = 10.
lam = K/N
dt = 0.1
sampling = 1
p =0.25

G = nx.erdos_renyi_graph(N, p2)
G = f.relabel_nodes(G)

nx.set_edge_attributes(G, lam, 'weight')
eigvals, _ = eigsh(nx.to_scipy_sparse_array(G).astype(float))
eig = np.min(eigvals)

sigma = -0.99/eig #high sigma as convergence might be tricky
psi = f.domirank(G, sigma=sigma)
attack = f.generate_attack(psi)

avg = 10



do = 'evolution'



r_total = np.zeros(steps)
for k in [4, 6, 10, 20]:
    p2 = k/(N-1)
    G = nx.erdos_renyi_graph(N, p2, seed=42)
    G = f.relabel_nodes(G)
    nx.set_edge_attributes(G, lam, 'weight')
 
    for lam in [2.0, 0.1, 0.5, 0.7, 1.]:
        t_init = time.time()
        for _ in range(avg):
            theta = fk.initialize_random(N)
            omegas = fk.initialize_random(N)
            r, _ = fk.evolve_kuramoto(theta, G, lam, omegas, dt, steps)
            r_total += np.array(r)
        r_total /= avg
        base_out = f'results_kuramoto/{net_name}_k{k}/dt{-np.log10(dt):.0f}_steps{steps}_avg{avg}/'
        if not os.path.exists(base_out):
            os.makedirs(base_out)
        out = f'{base_out}{do}_lam{lam:.1f}'
        x_axis = np.linspace(0, 1, len(r_total))
        plt.plot(x_axis, r_total, label = f"λ={lam:.1f}", linewidth=2)
        np.savetxt(out + '_results.txt', np.array(r_total), fmt='%.4f')
        np.savetxt(out + '_convergence.txt', np.array([time.time() - t_init, np.linalg.norm(r_total[-1]-r_total[-2])]), fmt='%.4f', delimiter='\t\t')
        
        print(f'lambda={lam} done')
    plt.xlabel("Time steps")
    plt.ylabel("Order parameter r")
    plt.legend()
    plt.savefig(base_out + f'{do}_r.png', bbox_inches='tight')
    plt.close()

    
    

    