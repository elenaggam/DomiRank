import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import functions as f
from scipy.sparse.linalg import eigsh
import functions_kuramoto as fk
import functions as f
from joblib import Parallel, delayed



net_name = 'ER'

N = 100
avg_k = 8
p2 = avg_k/(N-1)

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


avg=2
def run_kuramoto(dt):
    steps = 1000
    
    print(f'dt={dt} starting')
    for k in [4, 6,  10, 20]:
        p2 = k/(N-1)
        G = nx.erdos_renyi_graph(N, p2, seed=42)
        G = f.relabel_nodes(G)
        
        print(f'k={k} starting')
    
        for lam in np.flip(np.arange(0.1, 1.6, 0.1)):
            nx.set_edge_attributes(G, lam, 'weight')
            conv_iter = 0
            t_init = time.time()
            r_total = 0
            base_out = f'results_kuramoto/{net_name}_{N}/k{k}_dt{-np.log10(dt):.0f}_steps{steps}_avg{avg}/{do}/'
            if not os.path.exists(base_out):
                os.makedirs(base_out)
            
            # average over different initial conditions to remove fluctuations
            for _ in range(avg):
                theta = fk.initialize_random(N)
                omegas = fk.initialize_random(N)
                r, _, conv_step = fk.evolve_kuramoto(theta, G, omegas, dt, steps, break_on_convergence=False)
                print(conv_step)
                if _ == 0:
                    r_total = np.array(r)
                    
                r_total += np.array(r)
                if np.any(np.asarray(r) > 1.):
                    print(f'r integration not correct for lambda={lam:.1f}, dt={dt}, k={k}!')
                conv_iter += conv_step
            r_total /= avg
            conv_step = conv_iter/avg
            
            out = base_out + f'{lam:.1f}'

            # plotting the evolution of r
            
            done = time.time() - t_init
            
            np.savetxt(out + '_results.txt', np.array(r_total), fmt='%.4f')
            file = open(out + '_convergence.txt', 'w')
            file.write(f'runtime (s)\t\t{done:.0f}\nfinal r_err\t\t{np.linalg.norm(r_total[-1]-r_total[-2]):.5f}\nConvergence time\t\t{conv_step*dt:.2f}\n')
            file.close()

            print(f'lambda={lam} done in {done:.2f} seconds, convergence time: {conv_step*dt:.2f} time units, final r={r_total[-1]:.4f}')


    
    
for dt in [0.1, 0.01]:
    run_kuramoto(dt)
    