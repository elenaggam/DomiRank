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



# def sequential_recovery(G, p, attackStrategy, sampling = 0):

#     if type(G) == nx.classes.graph.Graph: #check if it is a networkx Graph
#         GAdj = nx.to_scipy_sparse_array(G) #convert to scipy sparse if it is a graph 
#     else:
#         GAdj = G.copy()

#     N = GAdj.shape[0]
#     initialComponent = float(get_component_size(GAdj)) # for normalization to lcc(0) = 1
#     initialLinks = float(get_link_size(G))

#     if sampling == 0: # sample every 1% of the nodes removed by default
#         sampling = int(N/100)
#         if sampling == 0: # if the graph is too small, we sample every node
#             sampling = 1
    
#     # evolution of the links and lcc, according to sampling
#     links = np.zeros(int(N/sampling)) 
#     component = np.zeros(int(N/sampling))

#     j = 0 # save data every N/sampling steps
#     for i in range(N-1):
#         if i%sampling == 0:
#             if i != 0: # avoid saving the initial condition twice
#                 to_remove = attackStrategy[i-sampling:i] # as we skipped sampling nodes, we remove the skipped nodes all at once
#                 for node in to_remove:
#                     if np.random.rand() > p: # with probability p, we recover the nodes (not remove them)
#                         print(f"deleted node {node} in iteration {i}")
#                         GAdj = remove_node(GAdj, node) 

#             links[j] = get_link_size(GAdj)/initialLinks # get the interest parameters (normalized)
#             component[j] = get_component_size(GAdj)/initialComponent
#             j += 1

#     return component, links

# def random_recovery(G, p, attackStrategy, sampling = 0):

#     if type(G) == nx.classes.graph.Graph: #check if it is a networkx Graph
#         GAdj_og = nx.to_scipy_sparse_array(G) #convert to scipy sparse if it is a graph 
#     else:
#         GAdj_og = G.copy()

#     N = GAdj_og.shape[0]
#     initialComponent = float(get_component_size(GAdj_og)) # for normalization to lcc(0) = 1
#     initialLinks = float(get_link_size(GAdj_og))

#     if sampling == 0: # sample every 1% of the nodes removed by default
#         sampling = int(N/100)
#         if sampling == 0: # if the graph is too small, we sample every node
#             sampling = 1
    
#     # evolution of the links and lcc, according to sampling
#     links = np.zeros(int(N/sampling)) 
#     component = np.zeros(int(N/sampling))

#     removed_nodes = []

#     GAdj = GAdj_og.copy()
#     j = 0 # save data every N/sampling steps
#     for i in range(N-1):
#         if i%sampling == 0:
#             if len(removed_nodes) > 0:
#                 for node in range(sampling): # sampling time steps have passed
#                     chosen = np.random.choice(removed_nodes) # choose a random node from the removed nodes
#                     if np.random.rand() < p: # with probability p, we recover the nodes (not remove them)
#                         removed_nodes.remove(chosen) # remove the chosen node from the removed nodes pool
#                 print(len(removed_nodes))
                
#                 GAdj = remove_node(GAdj_og, removed_nodes)
#             to_remove = attackStrategy[i-sampling:i] # as we skipped sampling nodes, we remove the skipped nodes all at once
#             removed_nodes.extend(to_remove) 
#             links[j] = get_link_size(GAdj)/initialLinks # get the interest parameters (normalized)
#             component[j] = get_component_size(GAdj)/initialComponent
#             j += 1

#     return component, links



from matplotlib.pylab import eig
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import functions as f
from scipy.sparse.linalg import eigsh

N = 100

centrality_titles = ['Domirank', 'Katz', 'Betweenness', 'Degree', 'Harmonic', 'Closeness', 'Eigenvector', 'Load', 'PageRank', 'Current-flow']
centrality_func = [f.domirank, nx.katz_centrality_numpy, nx.betweenness_centrality, nx.degree_centrality, nx.harmonic_centrality, nx.closeness_centrality, nx.eigenvector_centrality_numpy, nx.load_centrality, nx.pagerank, nx.current_flow_betweenness_centrality]

network_titles = ['WS', 'ER (high degree)', 'ER (low degree)', 'BA', 'RGG']
network_functions = [nx.watts_strogatz_graph, nx.erdos_renyi_graph, nx.erdos_renyi_graph, nx.barabasi_albert_graph, nx.random_geometric_graph]
network_args = [dict(n=N, k=4, p=0.12), dict(n=N, p=0.35),  dict(n=N, p=0.12), dict(n=N, m=6), dict(n=N, radius=np.sqrt((2*26)/(np.pi*N*(N-1))))]

plotting = [0.0, 0.08, 0.18, 0.28, 0.4] # p of the attack to plot
# ER 0.12, seed=23
# BA 1, seed=82

# graph 
avgN = 10
sampling = 1

time_file = open(f"{N}_sigma{avgN}_time.txt", "w")

# loop through the different networks and compute the optimal sigma and the attacks
for i in range(len(network_functions)):
    print(f"\n\nProcessing {network_titles[i]}...")

    start_time = time.time()

    base = f"{network_titles[i]}/{N}_sigma{avgN}_{sampling}/"
    if not os.path.exists(base):
        os.makedirs(base)

    figure = plt.figure()

    sigma_file = open(f"{base}sigma.txt", "w")
    avg_sigma = 0
    avg_eig =0
    for k in range(avgN):
        G = network_functions[i](**network_args[i])
        sparse_G = nx.to_scipy_sparse_array(G)
        G = f.relabel_nodes(G) # relabel the nodes to be from 0 to N-1 instead of tuples
        eigenvalues, _ = eigsh(nx.to_scipy_sparse_array(G).astype(float))
        eig = np.min(eigenvalues) # lambda_N
        sigma, _ = f.old_optimal_sigma(network_functions[i](**network_args[i]), endVal = eig)
        avg_sigma += sigma
        avg_eig += eig
        sigma_file.write(f"{sigma}\t{eig}\n")
    
    avg_eig /= avgN
    avg_sigma /= avgN
    sigma_file.write(f"{avg_sigma}\t{avg_eig}\n")
    sigma_file.close()

    G = network_functions[i](**network_args[i])
    G = f.relabel_nodes(G) # relabel the nodes to be from 0 to N-1 instead of tuples

    psi = f.domirank(G, sigma=avg_sigma) #compute the centrality measures
    attack = f.generate_attack(psi)
    lcc, links = f.network_attack_sampled(G, attack, sampling = sampling)
    plt.plot(np.linspace(0, 1, len(lcc)), lcc, label = "Domirank", color = "b", linestyle='dotted', linewidth=3, zorder=10)
    print(f"Domirank attack completed")

    for j in range(1, len(centrality_func)-1):
        psi = list(centrality_func[j](G).values())
        attack = f.generate_attack(psi)
        lcc, links = f.network_attack_sampled(G, attack, sampling = sampling)
        plt.plot(np.linspace(0, 1, len(lcc)), lcc, label = centrality_titles[j], linewidth=2)
        print(f"{centrality_titles[j]} attack completed")
    end_time = time.time()
    time_file.write(f"{network_titles[i]}: {end_time - start_time:.2f} seconds\n")
    
    plt.title(f"{network_titles[i]} σ={avg_sigma*avg_eig:.2f}/λ", fontsize=14)
    plt.legend(fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("removed fraction, p", fontsize = 12)
    plt.ylabel("Largest Connected Component", fontsize = 12)
    plt.savefig(base + "lcc_averaged.png", dpi=300, bbox_inches='tight')
    print(f"Plot saved for {network_titles[i]}")
    plt.close()


# for j in range(1, len(centrality_func)-1):
#         f.average_attack(N, network_functions[i], network_args[i], centrality_titles[j], centrality_func[j], base = base, avgN = avgN, sampling = sampling)
#         lcc, links = np.loadtxt(f"{base}{centrality_titles[j]}_averaged_lcc_links.txt", unpack = True)
#         plt.plot(np.linspace(0, 1, len(lcc)), lcc, label = centrality_titles[j], linewidth=2)
#         print(f"{centrality_titles[j]} attack completed")