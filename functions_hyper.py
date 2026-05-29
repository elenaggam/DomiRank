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
import functions as f


def dict_to_matrix(datos_dict):
    '''
    Makes incidence matrix from a dictionary of the form {edge: [nodes in the edge]}.
    '''
    aristas_nombres = list(datos_dict.keys())
    nodos_nombres = sorted(list(set(nodo for nodos in datos_dict.values() for nodo in nodos)))

    matriz_incidencia = np.zeros((len(nodos_nombres), len(aristas_nombres)))

    for j, arista in enumerate(aristas_nombres):
        for nodo in datos_dict[arista]:
            i = nodos_nombres.index(nodo)
            matriz_incidencia[i, j] = 1

    return matriz_incidencia, nodos_nombres, aristas_nombres

def func(const, eta, e):
    '''
    generalisation of theta and alpha for domirank
    '''
    if eta == 0:
        return np.full(e.shape, const, dtype=float)
    return const * (e - 1) ** eta

def domirank(H_matrix, alpha, theta, eta1, eta2, dt = 0.1, epsilon = 1e-5, maxIter = 10000, checkStep = 10):
    '''
    generalisation of domirank for hypergrphs, here, beta is set to 1
    '''
    # H_matrix = H.incidence_matrix().tocsr() # shape = (N, E)
    N, E = H_matrix.shape
    H_matrix_copy = H_matrix.copy()
    
    # hyperedge cardinality list
    e_list = np.array(H_matrix.sum(axis=0)).flatten() # sum along the node axis -> list of cardinalities, shape = (E,)
    alphas = func(alpha, eta1, e_list) 
    thetas = func(theta, eta2, e_list) 

    psi = np.zeros(N)
    boundary = epsilon*N

    for i in range(maxIter):
        tempVal = np.zeros(N)
        edge_contributions = np.zeros(E)
        auto_contributions = np.zeros(N)

        for edges in range(E): # iterate over edges
            alpha_e = alphas[edges]
            theta_e = thetas[edges]
            # scalar product of the edge contribution to each node in the edge
            # if a node does not belong to the edge, its contribution is zero as H_matrix[:, edges]=0 for that node
            edge_contributions[edges] = alpha_e * (theta_e - H_matrix_copy[:, edges].T@ psi)
            for node in range(len(H_matrix_copy[:, edges])): # iterate over nodes in the edge
                auto_contributions[node] += alpha_e*psi[node]*H_matrix_copy[node, edges]
            # OLD
            # edge_contributions[edges] = alpha_e * H_matrix_copy[:, edges].T@(theta_e - psi)
            # for node in range(len(H_matrix_copy[:, edges])): # iterate over nodes in the edge
            #     auto_contributions[node] += alpha_e*(theta_e-psi[node])*H_matrix_copy[node, edges]

        tempVal = H_matrix_copy @ edge_contributions - auto_contributions # shape = (N,)
        tempVal -= psi
        psi += tempVal.real*dt

        if i% checkStep == 0:
            if np.abs(tempVal).sum() < boundary:
                print(f"Converged at iteration {i}")
                # conv_iter = i
                break

    return psi

def plotting_psi(H, psi, node_names, title, layout='some', pos=None):
    '''
    helper function to plot the psi values on the hypergraph, with a color map
    '''
    psi_dict = dict(zip(node_names, psi/np.max(psi)))
    psi_ordered = [psi_dict[n] for n in H.nodes]

    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = cm.cividis
    node_colors = [cmap(norm(v)) for v in psi_ordered]

    G = H.bipartite()
    if pos is None:
        if layout == 'circular':
            pos = nx.circular_layout(G)
        else :
            pos = nx.spring_layout(G, seed=42) 

    fig, ax = plt.subplots(figsize=(8, 6))

    
    hnx.draw(H, ax=ax, pos=pos, nodes_kwargs={"facecolors": node_colors})

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    # plt.colorbar(sm, ax=ax, label="psi", shrink=0.5 )
    plt.title(title)
    return 

def prueba_step(H, matriz, nombres_nodos, alpha, theta, eta1, eta2, layout='some'):
    '''
    helper function to calculate domirank and plot the hypergraph
    '''
    psi = domirank(matriz, alpha, theta, eta1, eta2)
    title = f"α={alpha}(e-1)$^{{{eta1}}}$, θ={theta}(e-1)$^{{{eta2}}}$"
    plotting_psi(H, psi, nombres_nodos, title, layout=layout)

def attacks(H, H_matrix, attackStrategy, node_names, edge_names, psi, plot_to = 5, layout='some'):
    '''
    attacking a hypergraph sequentially according to the attack strategy, and plotting the evolution of psi on the hypergraph
    '''
    
    H_matrix_copy = H_matrix.copy()
    node_names_copy = node_names.copy()
    edge_names_copy = edge_names.copy()

    G = H.bipartite()
    if layout == 'circular':
            pos = nx.circular_layout(G)
    else :
        pos = nx.spring_layout(G, seed=42) 


    for i in range(len(attackStrategy)-1):
        removed_node = attackStrategy[i]
        removed_node_index = np.where(np.array(node_names_copy) == removed_node)[0][0] # find the index of the node to remove
        node_names_copy.pop(removed_node_index) # remove the node from the node names list
        H_matrix_copy = np.delete(H_matrix_copy, removed_node_index, axis=0) # remove the node from the incidence matrix
        
        # remove_col = []
        # for j in range(H_matrix_copy.shape[1]):
        #     if H_matrix_copy[:,j].sum() == 0: # if the edge has no nodes left, remove it
        #         print(f"Removing edge {H.edges[j]} as it has no nodes left at iteration {i}")
        #         remove_col.append(j)
        # H_matrix_copy = np.delete(H_matrix_copy, remove_col, axis=1) # remove the empty edges from the incidence matrix
        # edge_names_copy = np.delete(edge_names_copy, remove_col)

        H_copy = hnx.Hypergraph.from_numpy_array(
            H_matrix_copy,
            node_names=node_names_copy,
            edge_names=edge_names_copy      )
        if i<= plot_to:
            plotting_psi(H_copy, psi, node_names_copy, title=f"After removing node {attackStrategy[i]}", layout=layout, pos=pos)

    return

def domirank_save_evolution_theta(H_matrix, alpha, theta, eta1, eta2_list, file_title, dt = 0.1, epsilon = 1e-5, maxIter = 10000, checkStep = 1):
    '''
    function to save the evolution of domirank as theta changes
    '''
    
    # H_matrix = H.incidence_matrix().tocsr() # shape = (N, E)
    N, E = H_matrix.shape
    H_matrix_copy = H_matrix.copy()
    
    # hyperedge cardinality list
    e_list = np.array(H_matrix.sum(axis=0)).flatten() # sum along the node axis -> list of cardinalities, shape = (E,)
    alphas = func(alpha, eta1, e_list) 

    psi = np.zeros(N)
    boundary = epsilon*N
    file_convergence = open(file_title + '_convergence.txt', 'w')
    with open(file_title + '_evolution.txt', 'w') as f:
        for eta2 in eta2_list:
            thetas = func(theta, eta2, e_list)
            for i in range(maxIter):
                tempVal = np.zeros(N)
                edge_contributions = np.zeros(E)
                auto_contributions = np.zeros(N)

                for edges in range(E): # iterate over edges
                    alpha_e = alphas[edges]
                    theta_e = thetas[edges]
                    # scalar product of the edge contribution to each node in the edge
                    # if a node does not belong to the edge, its contribution is zero as H_matrix[:, edges]=0 for that node
                    edge_contributions[edges] = alpha_e * (theta_e - H_matrix_copy[:, edges].T@ psi)
                    for node in range(len(H_matrix_copy[:, edges])): # iterate over nodes in the edge
                        auto_contributions[node] += alpha_e*psi[node]*H_matrix_copy[node, edges]
                    # OLD
                    # edge_contributions[edges] = alpha_e * H_matrix_copy[:, edges].T@(theta_e - psi)
                    # for node in range(len(H_matrix_copy[:, edges])): # iterate over nodes in the edge
                    #     auto_contributions[node] += alpha_e*(theta_e-psi[node])*H_matrix_copy[node, edges]

                tempVal = H_matrix_copy @ edge_contributions - auto_contributions # shape = (N,)
                tempVal -= psi
                psi += tempVal.real*dt

                if i% checkStep == 0:
                    for p in psi:
                        f.write(f"{p:.4f}\t")
                    f.write("\n")

                    if np.abs(tempVal).sum() < boundary:
                        print(f"Converged at iteration {i}")
                        file_convergence.write(f"{eta2}\t{i}\n")
                        # conv_iter = i
                        break

    file_convergence.close()
    return 