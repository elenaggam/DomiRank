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

def correct_theta(theta):
    # we want to correct the theta values to be between -pi and pi, so we can visualize them better
    theta = np.mod(theta + np.pi, 2*np.pi) - np.pi
    return theta

def plot_weighted_graph(G, w, theta, r, removed):
    theta = correct_theta(theta)
    node_size = [300 if i not in removed else 100 for i in G.nodes()]
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_size=np.array(node_size), node_color = np.abs(theta-np.mean(theta)), cmap = plt.cm.cividis)
    nx.draw_networkx_edges(G, pos, alpha = [w[i]/lam for i in range(len(w))], width = [w[i]/lam*2 for i in range(len(w))], edge_color = 'gray')
    plt.axis('off')
    #plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.turbo), label='|θ - <θ>|')
    plt.title(f'r = {r:.2f} at {len(removed)}/{N} removed nodes')
    plt.savefig(f'{len(removed)}.png', dpi=300, bbox_inches='tight')
    plt.close()

def kuramoto_attack2(theta, omega, G, dt, steps, attackStrategy = [], see_at = [0, 5, 8, 10], sampling =0, sampling_kura = 0, epsilon = 1e-4, centrality_func=None):
    '''
    attack the network and compute the order parameter, links and component size after each attack step, 
    until we have attacked all the nodes in attackStrategy.
    '''
    if type(G) == nx.classes.graph.Graph: #check if it is a networkx Graph
        GAdj = nx.to_scipy_sparse_array(G) #convert to scipy sparse if it is a graph
    else:
        GAdj = G.copy()
    N = len(theta)
    rows, cols = GAdj.nonzero()
    w = GAdj.data

    if sampling == 0: # sample every 1% of the nodes removed by default
        sampling = int(N/100)
        if sampling == 0: # if the graph is too small, we sample every node
            sampling = 1

    if sampling_kura == 0:
        sampling_kura = int(N/100)
        if sampling_kura == 0: 
            sampling_kura = 1


    r = []
    r_before_attack = []
    node_map = {i: i for i in range(N)} # we need to keep track of the node labels as they change after each attack, as the attack strategy is based on the original labels
    r_temporal, theta, _ = fk.evolve_kuramoto(theta, omega, dt, rows, cols, w, steps=steps, epsilon=epsilon, sampling=sampling_kura)
    r_before_attack.append(r_temporal[-1]) # we save the order parameter before the attack to see the effect of the attack on the order parameter
    removed = []
    for i in range(N-1):
        # we evolve the Kuramoto model after each attack to see the effect on the order parameter and sinchronization
        if i%sampling == 0: # we also want to save the original state:
            if i != 0:
                attacked, node_map = f.changing_attack(GAdj, attackStrategy=attackStrategy, centrality_func=centrality_func, node_map=node_map, sampling=sampling, i=i) 
                w = fk.change_edge_weight(w, rows, cols, attacked, 0)
                r_more, theta, _ = fk.evolve_kuramoto(theta, omega, dt, rows = rows, cols = cols, w = w, steps = steps, epsilon = epsilon, sampling = sampling_kura)
                r.extend(r_more) 
                removed.append(attacked)
                r_before_attack.append(r_more[-1]) # we save the order parameter before the attack to see the effect of the attack on the order parameter      
        if i in see_at:
            plot_weighted_graph(G, w, theta, r_before_attack[-1], removed)
    return r, r_before_attack
 
N = 50
dt = 0.1
lam = 20.
removed_nodes = []
G, eig = make_graph('BA', N, 4)
psi = f.domirank(G, sigma = -0.99/eig)
attackStrategy = f.generate_attack(psi)
nx.set_edge_attributes(G, lam, 'weight')
theta = fk.initialize_random(N)
omegas = fk.initialize_random(N)
pos = nx.spring_layout(G, iterations=500, seed=160)

r, r_before_attack = kuramoto_attack2(theta, omegas, G, dt, 1000, attackStrategy=attackStrategy, sampling=1, sampling_kura=10) 

plt.plot(r_before_attack, label='Order Parameter r')
plt.show()


