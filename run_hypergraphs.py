import hypernetx as hnx
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import functions as f
import functions_hyper as fh



hg_1 = {
    1: [1,2, 3, 4, 5],
    2: [12, 13],
    3: [6, 7],
    4: [1,8, 9],
    5: [11, 12, 10, 2],
    6: [7, 9]
}

hg_2 = {
    1: [1,8, 7, 10, 11],
    2: [1, 2],
    3: [1, 3],
    4: [4, 5, 6],
    5: [6, 7],
    6: [8, 9],
    7: [2, 12],
    8: [4, 13, 14, 15],
}

hg_3 = {
    0: [1, 2, 3, 4, 5],
    1: [1, 6],
    2: [2, 7],
    3: [3, 8],
    4: [4, 9],
    5: [5, 10, 11,12],
    6: [6, 13, 14],
    7: [7, 15, 19, 20, 21], 
    8: [8, 16],
    9: [9, 17, 18],   
            }

hg_4 = {
    0: [1, 2],
    1: [2, 3, 4],
    2: [4, 5, 6, 7],
    3: [7, 8, 9, 10, 11],
    4: [11, 12, 13, 14, 15, 16],
    5: [15, 16, 17], 
    6: [17, 18, 19],
    7: [20, 19],
}

hg_5 = {
    0: [1, 2, 3],
    1: [3, 4],
    2: [5, 6],
    3: [6, 7, 8],
    4: [4, 5, 9, 10], 
    5: [9, 10, 11], 
    6: [11, 12]
}

alpha = 0.05
theta = 2.
eta2 = 0
eta = 1.3

matriz, nombres_nodos, nombres_aristas = fh.dict_to_matrix(hg_3)
H = hnx.Hypergraph.from_numpy_array(
    matriz, 
    node_names=nombres_nodos, 
    edge_names=nombres_aristas)

# dr = fh.domirank(matriz, alpha, theta, eta, eta2)
# for seed in range(200):
#     pos = nx.spring_layout(H.bipartite(), seed=seed)
#     fh.plotting_psi(H, dr, nombres_nodos, title=f"Seed {seed}", layout='some', pos=pos)
#     plt.savefig(f'results_hypernet/pruebasseed/net3_alpha{alpha}_theta{theta}_eta1_{eta}_eta2_{eta2}_seed{seed}.png')
#     plt.close()



for eta in [-2, 0, 2]:
    for eta2 in [-2, 0, 2]:
        fh.prueba_step(H, matriz, nombres_nodos, alpha, theta, eta, eta2)
        plt.savefig(f'results_hypernet/pruebas2/net3_alpha{alpha}_theta{theta}_eta1_{eta}_eta2_{eta2}.png')
        plt.close()

# psi = fh.domirank(matriz, alpha, theta, eta, eta2)

# node_map = {i: nombre for i, nombre in enumerate(nombres_nodos)}
# attack = f.generate_attack(psi, node_map)
# fh.plotting_psi(H, psi, nombres_nodos, title = ' ', layout='circular')
# fh.attacks(H, matriz, attack, nombres_nodos, nombres_aristas, psi, plot_to = 3, layout='circular')
# plt.show()
# input("Press Enter to continue...")