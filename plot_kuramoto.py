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
avg = 2

interval = 400

for k in [4, 6, 10, 20]:
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
plt.title(f"ER (N={N})")
plt.legend()
plt.savefig(f'results_kuramoto/{net_name}_{N}/evolution_r.png', dpi=300)
plt.close()