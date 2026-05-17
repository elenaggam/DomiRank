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


N = 20
p =0.5

net_name = 'BA'
k = 20
G = nx.barabasi_albert_graph(N, int(k/2), seed=42)
G = f.relabel_nodes(G)



def evolve(dt, k_list, avg_, list_lambdas=np.flip(np.arange(0.1, 1.6, 0.1))):
    steps = 100
    
    print(f'dt={dt} starting')
    for k in k_list:
        G = nx.barabasi_albert_graph(N, int(k/2), seed=42)
        G = f.relabel_nodes(G)
        
        print(f'k={k} starting')
    
        for lam in list_lambdas:
            nx.set_edge_attributes(G, lam, 'weight')
            conv_iter = 0
            t_init = time.time()
            if k<=6:
                avg = avg_[1]
            else:
                avg = avg_[0]
            r_total = 0
            base_out = f'results_kuramoto/{net_name}_{N}/k{k}_dt{-np.log10(dt):.0f}_steps{steps}_avg{avg}/attack/'
            if not os.path.exists(base_out):
                os.makedirs(base_out)
            out = base_out + f'{lam:.1f}'
            if os.path.isfile(out + '_results.txt'):
                print(f'lambda={lam:.1f} already done, skipping')
                continue
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
            
            

            # plotting the evolution of r
            
            done = time.time() - t_init
            
            np.savetxt(out + '_results.txt', np.array(r_total), fmt='%.4f')
            file = open(out + '_convergence.txt', 'w')
            file.write(f'runtime (s)\t\t{done:.0f}\nfinal r_err\t\t{np.linalg.norm(r_total[-1]-r_total[-2]):.5f}\nConvergence time\t\t{conv_step*dt:.2f}\n')
            file.close()

            print(f'lambda={lam} done in {done:.2f} seconds, convergence time: {conv_step*dt:.2f} time units, final r={r_total[-1]:.4f}')


def attack_or_recover(G, directory, sigma, strategy_name, do = 'attack', p =0.25, dt = 0.01, steps = 100, avg = 10, list_lambdas=np.flip(np.arange(0.1, 1.6, 0.1)), sampling = 0, sampling_kura = 0, centrality_func = None):
    
    base_out = f'{directory}dt{-np.log10(dt):.0f}_steps{steps}_avg{avg}/{do}/'
    steps = int(100/dt)
    if not os.path.exists(base_out):
        os.makedirs(base_out)
    base_out += strategy_name + '_'

    areas = []
    t_init = time.time()
    for lam in list_lambdas:
        out = base_out + f'{lam:.1f}'
        if os.path.isfile(out + '_results.txt'):
            print(f'lambda={lam:.1f} already done, skipping')
            continue

        nx.set_edge_attributes(G, lam, 'weight')
        
        for _ in range(avg):
            theta = fk.initialize_random(N)
            omega = fk.initialize_random(N)
            psi = f.domirank(G, sigma)
            attackStrategy = f.generate_attack(psi)

            if do == 'attack':
                r, r_before_attack = fk.kuramoto_attack(theta, G, omega, dt, steps, attackStrategy, sampling = sampling, sampling_kura = sampling_kura, centrality_func=centrality_func)
            elif do == 'recover':
                r, r_before_attack = fk.kuramoto_attack_recovery(theta, G, lam, omega, dt, steps, p=p, attackStrategy = attackStrategy, sampling = sampling, sampling_kura = sampling_kura, centrality_func=centrality_func)
            else: 
                raise ValueError('do must be either attack or recover')
            
            if _ == 0:
                r_total = np.array(r)
                r_before = np.array(r_before_attack)
            else:    
                r_total += np.array(r)
                r_before += np.array(r_before_attack)
            areas.append(fk.calculate_area(r, dt))

            if np.any(np.asarray(r) > 1.):
                print(f'r integration not correct for lambda={lam:.1f}, dt={dt}!')
        
        r_total /= avg
        r_before /= avg

        np.savetxt(out + '_results.txt', r_before, fmt='%.4f', delimiter = '\t')
        np.savetxt(out + '_r_evolution.txt', r, fmt='%.4f')

    np.savetxt(f'{base_out}areas.txt', np.column_stack((list_lambdas, areas)), fmt='%.4f', delimiter = '\t')

    t_final = time.time() - t_init
    np.savetxt(f'{base_out}runtime.txt', np.array([t_final]), fmt='%.2f')
    print(f'All lambdas done in {t_final:.2f} seconds')


eigvals, eigvecs = eigsh(nx.to_scipy_sparse_array(G).astype(float), k=1, which='LA')
eig = np.min(eigvals)
attack_or_recover(G, directory=f'results_kuramoto/{net_name}_{N}_k{k}/', sigma=-0.5/eig, strategy_name='domirank', do = 'recover', p =p, steps = 100, avg = 1, sampling = 1, sampling_kura = 1)