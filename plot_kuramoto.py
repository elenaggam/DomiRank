import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import functions as f
from scipy.sparse.linalg import eigsh
import functions_kuramoto as fk
import functions as f
from run_kuramoto import make_graph



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
N = 1000
steps = N
k = 6
p = 0.25
avg =8

sigmas = np.arange(0.03, 1.02, 0.03)
list_lambdas=np.arange(0.2, 2.05, 0.05)

do = [f'recover_sequential_p{p}', 'attack']

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
    plt.legend(fontsize=12)
    plt.xlabel('p', fontsize=14)
    plt.ylabel('LCC', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
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
                plt.plot(np.linspace(0, 1, len(r)), r, label=f'σ={sigma:.2f}')
            #areas2d[i] = np.loadtxt(f'{base}{d}/domirank_s{sigma:.2f}/' + f'areas_stable.txt', usecols=(1,))
            plt.xlabel('fracción de la simulación', fontsize=14)
            plt.ylabel('r', fontsize=14)
            plt.ylim(0, 1)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend(fontsize=12)
            plt.title(f'{d} - λ={lam:.2f} - {net_name} (N={N})')
            i+=1
            plt.savefig(f'{N}_{net_name}_{d}_r_stable_lambda{lam:.2f}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        #plot_heatmap(areas2d)
