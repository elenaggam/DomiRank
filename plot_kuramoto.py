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

net_name = 'ER'
dt = 0.1
N = 100
steps = 100
avg_ = [10, 10]
k = 6
do = 'attack'
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

area = []
lams = np.flip(np.arange(0.1, 2.1, 0.05))
sigmas = np.arange(0.01, 1, 0.03)
r_tot = []

for sigma in [0.16, 0.97]:
    a = np.loadtxt(f'results_kuramoto/{net_name}_{N}_k{k}/dt1_steps100_avg15_{do}/domirank_s{sigma:.2f}/areas.txt', usecols=(1,))
    area.append(np.flip(a))

plt.plot(lams, area[0], label=f'sigma={0.16:.2f}', marker='o')
plt.plot(lams, area[1], label=f'sigma={0.97:.2f}', marker='o')
plt.xlabel("λ")
plt.ylabel("Area")
plt.title(f"{net_name} (N={N})")
plt.legend()
plt.savefig(f'results_kuramoto/{net_name}_{N}_k{k}_area_{do}.png', dpi=300)
plt.close()

areas_2d = np.array(area)

plt.figure(figsize=(10, 6))
plt.imshow(areas_2d, aspect='auto', origin='lower', extent=[lams.min(), lams.max(), sigmas.min(), sigmas.max()], cmap='viridis')
plt.colorbar(label='Area')
plt.xlabel('λ')
plt.ylabel('σ')
plt.title(f'Heatmap - {net_name} (N={N})')
plt.savefig(f'results_kuramoto/{net_name}_{N}_k{k}_heatmap_areas_{do}.png', dpi=300, bbox_inches='tight')
plt.close()


plt.plot
