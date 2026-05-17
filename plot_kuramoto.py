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
dt = 0.1
N = 100
steps = 1000
avg_ = [8, 2]

interval = 500


for dt in [0.1]:

    for k in [20, 10, 6, 4]:
            
        if k<=6:
            avg = avg_[0]
        else:            
            avg = avg_[1]
        r_points = []
        r_error = []
        x_axis = np.arange(0.1, 1.6, 0.1)
        for lam in np.arange(0.1, 1.6, 0.1):
            if k<=6:
                ra = np.loadtxt(f'results_kuramoto/{net_name}_{N}/k{k}_dt{-np.log10(dt):.0f}_steps{steps}_avg6/evolution/{lam:.1f}_results.txt')
                rb = np.loadtxt(f'results_kuramoto/{net_name}_{N}/k{k}_dt{-np.log10(dt):.0f}_steps{steps}_avg2/evolution/{lam:.1f}_results.txt')
                r = (ra + rb) / 2
                os.makedirs(f'results_kuramoto/{net_name}_{N}/k{k}_dt{-np.log10(dt):.0f}_steps{steps}_avg{avg}/evolution/', exist_ok=True)
                np.savetxt(f'results_kuramoto/{net_name}_{N}/k{k}_dt{-np.log10(dt):.0f}_steps{steps}_avg{avg}/evolution/{lam:.1f}_results.txt', r)
            else:
                base_out = f'results_kuramoto/{net_name}_{N}/k{k}_dt{-np.log10(dt):.0f}_steps{steps}_avg{avg}/evolution/'
                r = np.loadtxt(f'{base_out}{lam:.1f}_results.txt')
            r_points.append(np.mean(r[interval:])) # we only plot the second half of the evolution to better see the convergence
            r_error.append(np.std(r[interval:])) # we also plot the standard deviation to show the fluctuations
        plt.errorbar(x_axis, r_points, yerr=r_error, label = f"<k>={k}", marker='o',   linewidth=2)
    plt.xlabel("λ")
    plt.ylabel("Order parameter r")
    plt.title(f"ER (N={N})")
    plt.legend()
    plt.savefig(f'results_kuramoto/{net_name}_{N}/dt{-np.log10(dt):.0f}_evolution_r.png', dpi=300)
    plt.close()