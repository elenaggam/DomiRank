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
avg = 10


for k in [4, 6, 8, 10, 20]:
    r_points = []
    for lam in [0.1, 0.5, 0.7, 1., 2.]:
        x_axis = np.linspace(0, 1, steps)
        base_out = f'results_kuramoto/{net_name}_k{k}/dt{-np.log10(dt):.0f}_steps{steps}_avg{avg}/'
        r = np.loadtxt(f'{base_out}evolution_lam{lam:.1f}_results.txt')
        plt.plot(x_axis, r, label = f"λ={lam:.1f}", linewidth=2)
    plt.xlabel("time")
    plt.ylabel("Order parameter r")
    plt.title(f"<k>={k}")
    plt.legend()
    plt.savefig(f'results_kuramoto/{net_name}_k{k}/dt{int(-np.log10(dt))}_steps{steps}_avg{avg}/evolution_r.png', dpi=300)
    plt.close()