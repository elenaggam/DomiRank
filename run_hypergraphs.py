import hypernetx as hnx
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import functions as f
import functions_hyper as fh



hg_1 = {
    1: ['A','B', 'C', 'D', 'E'],
    2: ['L', 'M'],
    3: ['F', 'G'],
    4: ['A','H', 'I'],
    5: ['K', 'L', 'J', 'B'],
    6: ['G', 'I']
}

hg_2 = {
    1: ['A','H', 'G', 'J', 'K'],
    2: ['A', 'B'],
    3: ['A', 'C'],
    4: ['D', 'E', 'F'],
    5: ['F', 'G'],
    6: ['H', 'I'],
    7: ['B', 'L'],
    8: ['D', 'M', 'N', 'O'],
}

hg_3 = {
    0: ['A','B', 'C', 'D', 'E'],
    1: ['A', 'F'],
    2: ['C', 'H'],
    3: ['D', 'I'],
    4: ['E', 'J'],
    5: ['F', 'K', 'L', 'M'],
    6: ['B', 'N', 'O'],
    7: ['H', 'P', 'X', 'Y', 'Z'], 
    8: ['I', 'Q'],
    9: ['J', 'R', 'S'],   
            }


alpha = 0.05
theta = 2.
eta2 = 0
eta = -1.3

matriz, nombres_nodos, nombres_aristas = fh.dict_to_matrix(hg_3)
H = hnx.Hypergraph.from_numpy_array(
    matriz, 
    node_names=nombres_nodos, 
    edge_names=nombres_aristas)



for eta in np.arange(-3., 3.5, 0.5):
    fh.prueba_step(H, matriz, nombres_nodos, alpha, theta, eta, eta2)
    plt.savefig(f'net3_alpha{alpha}_theta{theta}_eta1_{eta}.png')
    plt.close()

# psi = fh.domirank(matriz, alpha, theta, eta, eta2)

# node_map = {i: nombre for i, nombre in enumerate(nombres_nodos)}
# attack = f.generate_attack(psi, node_map)
# fh.plotting_psi(H, psi, nombres_nodos, title = ' ')
# fh.attacks(H, matriz, attack, nombres_nodos, nombres_aristas, psi, plot_to = 3)
# plt.show()