from matplotlib import cm
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.sparse
from scipy.sparse.linalg import eigsh
import hypernetx as hnx
import matplotlib.colors as mcolors
import matplotlib.cm as cm


def dict_to_matrix(datos_dict):
    aristas_nombres = list(datos_dict.keys())
    nodos_nombres = sorted(list(set(nodo for nodos in datos_dict.values() for nodo in nodos)))

    matriz_incidencia = np.zeros((len(nodos_nombres), len(aristas_nombres)))

    for j, arista in enumerate(aristas_nombres):
        for nodo in datos_dict[arista]:
            i = nodos_nombres.index(nodo)
            matriz_incidencia[i, j] = 1

    return matriz_incidencia, nodos_nombres, aristas_nombres

def func(const, eta, e):
    if eta == 0:
        return np.full(e.shape, const, dtype=float)
    return const * (e - 1) ** eta

def domirank(H_matrix, alpha, theta, eta1, eta2, dt = 0.1, epsilon = 1e-5, maxIter = 10000, checkStep = 10):
    
    # H_matrix = H.incidence_matrix().tocsr() # shape = (N, E)
    N, E = H_matrix.shape
    
    # hyperedge cardinality list
    e_list = np.array(H_matrix.sum(axis=0)).flatten() # sum along the node axis -> list of cardinalities, shape = (E,)
    alphas = func(alpha, eta1, e_list) 
    thetas = func(theta, eta2, e_list) 

    psi = np.zeros(N)
    boundary = epsilon*N*dt

    for i in range(maxIter):
        tempVal = np.zeros(N)

        for node in range(N): # iterate over nodes
            edges = np.where(H_matrix[node, :] == 1)[0] # get edges containing the node
            for j in edges: # iterate over edges containing the node
                alpha_e = alphas[j]
                theta_e = thetas[j]
                tempVal[node] += alpha_e * H_matrix[node, j] * (theta_e - psi[node]) - psi[node]
            
        psi += tempVal.real*dt
        if i% checkStep == 0:
            if np.abs(tempVal).sum() < boundary:
                # print(f"Converged at iteration {i}")
                # conv_iter = i
                break

    return psi

def plotting_psi(H, psi, node_names, title):
    psi_dict = dict(zip(node_names, psi/np.max(psi)))
    psi_ordered = [psi_dict[n] for n in H.nodes]

    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = cm.cividis
    node_colors = [cmap(norm(v)) for v in psi_ordered]

    fig, ax = plt.subplots(figsize=(8, 6))

    hnx.draw(H, ax=ax, nodes_kwargs={"facecolors": node_colors})

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    plt.colorbar(sm, ax=ax, label="psi", shrink=0.5 )
    plt.title(title)
    plt.show()

datos = {
    1: ['A','B', 'C', 'D', 'E'],
    2: ['F', 'D'],
    3: ['F', 'G'],
    4: ['A','H', 'I'],
    5: ['J', 'I'],
    6: ['K', 'L', 'J', 'B']
}

matriz, nombres_nodos, nombres_aristas = dict_to_matrix(datos)

H = hnx.Hypergraph.from_numpy_array(
    matriz, 
    node_names=nombres_nodos, 
    edge_names=nombres_aristas
)

alpha = 0.5
theta = 0.5
eta1 = 1
eta2 = 3
psi = domirank(matriz, alpha, theta, eta1, eta2)
title = f"α={alpha:.1f}(e-1)$^{{{eta1}}}$, θ={theta:.1f}(e-1)$^{{{eta2}}}$"
plotting_psi(H, psi, nombres_nodos, title)
plt.savefig("hyperdomirank2_plot.png", dpi=300, bbox_inches='tight')