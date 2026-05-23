import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import functions as f
from scipy.sparse.linalg import eigsh
import functions_kuramoto as fk
import functions as f
from matplotlib.colors import LogNorm
from run_kuramoto import make_graph

def save_r_stable(r):
    r_out = []
    r_mean = 0
    r_averaged = []
    size = 0
    for i in range(1, len(r)):
        if np.linalg.norm(r[i] - r[i-1]) < 1e-4 and r[i] > 0.2:
            r_out.append(r[i])
    for i in range(1,len(r_out)):
        if np.linalg.norm(r_out[i] - r_out[i-1]) < 1e-4:
            r_mean += r_out[i-1]
            size += 2
        else:
            r_mean += r_out[i]
            r_averaged.append(r_mean / size if size > 0 else 0)
            r_mean = 0
            size = 0
    return r_out

def reshape_r_stable(r, n=100):
    groups = np.array_split(r, n)

    new_points = np.array([
    g.mean(axis=0)
    for g in groups
    ])

    return new_points

def plot_heatmap(areas2d):
    plt.imshow(areas2d, cmap='inferno', aspect='auto', extent=[list_lambdas[0], list_lambdas[-1], sigmas[0], sigmas[-1]], origin='lower')
    #plt.contour(list_lambdas, sigmas, areas2d, colors='white', linewidths=0.7, levels=5)
    plt.xlabel("λ")
    plt.ylabel("σ")
    plt.title(f"{d} - {net_name} (N={N})")
    plt.colorbar(label='Area')
    plt.savefig(f'{base}{d}/_area_heatmap.png', dpi=300)
    plt.close()
    return



dt = 0.1
N = 100
steps = N
k = 6
p = 0.25
avg =11

sigmas = np.arange(0.03, 1.02, 0.03)
list_lambdas=np.arange(0.2, 2.05, 0.05)

do = [f'recover_random_p{p}']

sigmas = [0.21, 0.99]
list_lambdas = [0.2, 1.0]



for net_name in ['ER', 'BA']:
    base = f'results_kuramoto/_{net_name}_{N}_k{k}/dt1_steps{steps}_avg{avg}_'

    G, eig = make_graph(net_name, N, k)
    optimal, _ =f.old_optimal_sigma(nx.to_scipy_sparse_array(G), eig)
    sigmas = [0.21, 0.99, -optimal*eig]
    for sigma in sigmas:
        psi = f.domirank(G, sigma=-sigma/eig)
        attack = f.generate_attack(psi)
        lcc, _ = f.network_attack_sampled(G, attack)

        plt.plot(lcc, label=f'σ={sigma:.2f}')
    plt.legend()
    plt.savefig(f'{N}_{net_name}_s{sigmas[-1]:.2f}_lcc_attack.png', dpi=300, bbox_inches='tight')
    plt.close()

    for d in do:
        areas2d = np.zeros((len(sigmas), len(list_lambdas)))
        #plot_heatmap()
        i=0
        for lam in list_lambdas:
            for sigma in sigmas:
                path = f'{base}{d}/domirank_s{sigma:.2f}/' + f'{lam:.3f}_r_stable.txt'
                if not os.path.exists(path):
                    continue
                r = np.loadtxt(f'{base}{d}/domirank_s{sigma:.2f}/' + f'{lam:.3f}_r_stable.txt', usecols=(0,))
                plt.plot( r, label=f'σ={sigma:.2f}')
            #areas2d[i] = np.loadtxt(f'{base}{d}/domirank_s{sigma:.2f}/' + f'areas_stable.txt', usecols=(1,))
            plt.xlabel('steps of simulation')
            plt.ylabel('Stable Synchronization (r)')
            plt.ylim(0, 1)
            plt.legend()
            plt.title(f'{d} - λ={lam:.2f} - {net_name} (N={N})')
            i+=1
            plt.savefig(f'{N}_{net_name}_{d}_r_stable_lambda{lam:.2f}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        #plot_heatmap(areas2d)