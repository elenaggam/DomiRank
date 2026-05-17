from os import path

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import functions as f
from scipy.sparse.linalg import eigsh
import functions_kuramoto as fk
import functions as f

net_name = 'BA'
dt = 0.1
N = 100
steps = 100
avg_ = [8, 2]

interval = 500


def plot_kuramoto_evolution():
    for dt in [0.01]:

        for k in [20, 10, 6, 4]:
                
            if k<=6:
                avg = avg_[0]
            else:            
                avg = avg_[1]
            r_points = []
            r_error = []
            x_axis = np.arange(0.1, 1.6, 0.1)
            for lam in np.arange(0.1, 1.6, 0.1):
                base_out = f'results_kuramoto/{net_name}_{N}/k{k}_dt{-np.log10(dt):.0f}_steps{steps}_avg{avg}/evolution/'
                r = np.loadtxt(f'{base_out}{lam:.1f}_results.txt')
                r_points.append(np.mean(r[interval:])) # we only plot the second half of the evolution to better see the convergence
                r_error.append(np.std(r[interval:])) # we also plot the standard deviation to show the fluctuations
            plt.errorbar(x_axis, r_points, yerr=r_error, label = f"<k>={k}", marker='o',   linewidth=2)
        plt.xlabel("λ")
        plt.ylabel("Order parameter r")
        plt.title(f"{net_name} (N={N})")
        plt.legend()
        plt.savefig(f'results_kuramoto/{net_name}_{N}/dt{-np.log10(dt):.0f}_evolution_r.png', dpi=300)
        plt.close()


for l in [1.5]:
    r = np.loadtxt(f'results_kuramoto/BA_20_k20/dt2_steps100_avg1/recover/domirank_{l:.1f}_r_evolution.txt')
    plt.plot(r, label = f"λ={l:.1f}")
    
plt.xlabel("Time step")
plt.ylabel("Order parameter r")
plt.legend(loc = 'upper left', bbox_to_anchor=(1, 1))
plt.savefig(f'results_kuramoto/BA_20_k20/dt2_steps100_avg1/recover/domirank_r_evolution.png', dpi=300, bbox_inches='tight')
plt.close()
base = 'results_kuramoto/BA_20_k20/dt2_steps100_avg1/recover/domirank_areas.txt'
plt.plot(np.loadtxt(base)[:,0], np.loadtxt(base)[:,1], marker='o')
plt.savefig('results_kuramoto/BA_20_k20/dt2_steps100_avg1/recover/domirank_area.png', dpi=300, bbox_inches='tight')
plt.close()

l = 1.5
r = np.loadtxt(f'results_kuramoto/BA_20_k20/dt2_steps100_avg1/recover/domirank_{l:.1f}_results.txt')
plt.plot(np.linspace(0, 1, len(r)) ,r, label = f"λ={l:.1f}")
plt.xlabel("removed nodes")
plt.ylabel("Order parameter r")
plt.title(f"{net_name} (N={N}) λ={l:.1f}")
plt.savefig(f'results_kuramoto/BA_20_k20/dt2_steps100_avg1/recover/domirank_r_removed_nodes.png', dpi=300, bbox_inches='tight')
plt.close()