from matplotlib.transforms import ScaledTranslation
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
import scipy
import os
import time

def domirank(G, sigma = -1, dt = 0.1, maxsteps = 10000, epsilon = 1e-5):
    '''
    Numerical (recursive) solution of the DomiRank centrality of the nodes in the graph G. 
    From the methods section in "DomiRank Centrality: revealing structural fragility 
    of complex networks via node dominance"
    
    Parameters:
    ------------
    G: a networkx graph
    sigma: float, optional (default = -1).
        The parameter of the DomiRank centrality. If -1, it is set to the upper limit of the convergence interval, which is -1/min(eigenvalues(G))
    dt: float, optional (default = 0.1).
        The time step of the numerical solution
    maxsteps: float, optional (default = 1e3).
        The maximum number of steps of the numerical solution
    epsilon: float, optional (default = 1e-5).
        The convergence threshold of the numerical solution. The convergence condition is ||gamma - gamma_prev|| < epsilon*N*dt, where N is the number of nodes in the graph
    method: str, optional (default = "euler").
        The numerical method to use. It can be "euler" or "heun". Heun's method is more accurate but slower than Euler's method.

    Returns:
    ------------
    gamma: numpy array of shape (N,) 
        Normalized DomiRank centrality of the nodes in the graph G. 
    '''
    
    N = G.number_of_nodes()

    if type(G) == nx.classes.graph.Graph: #check if it is a networkx Graph
        G = nx.to_scipy_sparse_array(G).astype(float) #convert to scipy sparse if it is a graph 
    else:
        G = G.copy()
        
    gamma = np.zeros(N, dtype = float)

    # if sigma is not provided, we set it to the upper limit of the convergence interval
    if sigma == -1: 
        eigenvalues, _ = eigsh(G)
        min = np.min(eigenvalues)
        del eigenvalues, _
        if min == 0:
            print("Warning in domirank: The maximum eigenvalue of the graph is zero. Returning zero vector.")
            return gamma
        sigma = -1/min

    # the convergence condition is ||gamma - gamma_prev|| < epsilon*N*dt
    convergence = epsilon*N*dt 
    conv_iter = 0
    for i in range(maxsteps):
        aux = (sigma*G @ (1-gamma) - gamma)*dt #f(t, gamma(t))*dt
        
        gamma += aux #gamma(t+dt) = gamma(t) + dt*f(t, gamma(t))

        if np.abs(aux).sum() < convergence:
            #print(f'Convergence reached after {i} steps.')
            conv_iter = i
            break
    if conv_iter == 0:
        conv_iter = maxsteps
    return gamma/np.max(gamma), conv_iter #normalize to [0,1]

# def old_optimal_sigma(G, delta_sigma = 0.001, sampling = 0, dt = 0.1, epsilon = 1e-5, maxIter = 10000, checkStep = 10):
    
#     if type(G) == nx.classes.graph.Graph: #check if it is a networkx Graph
#         GAdj = nx.to_scipy_sparse_array(G).astype(float) #convert to scipy sparse if it is a graph 
#     else:
#         GAdj = G.copy()
    
#     eig = eigsh(GAdj, return_eigenvectors=False) # get the largest eigenvalue of the adjacency matrix
#     sigma_max = -0.9999/np.min(eig)
#     sigma_range = np.arange(0.001, sigma_max, delta_sigma) 

#     optimal_sigma = -1.
#     min_lcc = -1.

#     for sigma in sigma_range:
#         Psi = domirank(GAdj, sigma = sigma, dt = dt, epsilon = epsilon, maxIter = maxIter, checkStep = checkStep)
#         attack = generate_attack(Psi)
#         lcc, _ = network_attack_sampled(GAdj, attack, sampling = sampling) # get the lcc after attacking with the generated attack strategy
#         area_lcc = np.sum(lcc)
#         if area_lcc < min_lcc or min_lcc == -1:
#             min_lcc = area_lcc
#             optimal_sigma = sigma
        
#     return optimal_sigma
