import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import functions as f
from scipy.sparse.linalg import eigsh

N = 1000

centrality_titles = ['Domirank', 'Katz', 'Betweenness', 'Degree', 'Harmonic', 'Closeness', 'Eigenvector', 'Load', 'PageRank', 'Current-flow']
centrality_func = [f.domirank, nx.katz_centrality_numpy, nx.betweenness_centrality, nx.degree_centrality, nx.harmonic_centrality, nx.closeness_centrality, nx.eigenvector_centrality_numpy, nx.load_centrality, nx.pagerank]

# network_titles = ['WS', 'ER (high degree)', 'ER (low degree)', 'BA', 'RGG']
# network_functions = [nx.watts_strogatz_graph, nx.erdos_renyi_graph, nx.erdos_renyi_graph, nx.barabasi_albert_graph, nx.random_geometric_graph]
# network_args = [dict(n=N, k=4, p=0.12), dict(n=N, p=20.0/(N-1.0)),  dict(n=N, p=6.0/(N-1.0)), dict(n=N, m=3), dict(n=N, radius=np.sqrt(16/(np.pi*N)))]

network_titles = ['connected WS']
network_functions = [nx.connected_watts_strogatz_graph]
network_args = [dict(n=N, k=4, p=0.12)]

# graph 
avgN = 1
sampling = 50

time_file = open(f"{N}_time.txt", "w")

# loop through the different networks and compute the optimal sigma and the attacks
for i in range(len(network_functions)):
    print(f"\n\nProcessing {network_titles[i]}...")

    start_time = time.time()

    G = network_functions[i](**network_args[i])
    sparse_G = nx.to_scipy_sparse_array(G)
    G = f.relabel_nodes(G) # relabel the nodes to be from 0 to N-1 instead of tuples
    eigenvalues, _ = eigsh(nx.to_scipy_sparse_array(G).astype(float))
    eig = np.min(eigenvalues) # lambda_N

    base = f"{network_titles[i]}/{N}_sigma{avgN}_s{sampling}/"
    if not os.path.exists(base):
        os.makedirs(base)

    figure = plt.figure()

    sigma = 0
    for a in range(avgN):
        H = network_functions[i](**network_args[i])
        H = f.relabel_nodes(H) # relabel the nodes to be from 0
        eigenvaluesh, _ = eigsh(nx.to_scipy_sparse_array(H).astype(float))
        eigh = np.min(eigenvaluesh) # lambda_N
        sigma_h, _ = f.old_optimal_sigma(G, endVal = eigh, directory = base)
        sigma += sigma_h
    sigma /= avgN
    print(f"Optimal sigma: {sigma*eig:.4f}/λ")
    psi = f.domirank(G, sigma=sigma) #compute the centrality measures
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
        del psi, attack, lcc, links
    
    end_time = time.time()
    time_file.write(f"{network_titles[i]}: {end_time - start_time:.2f} seconds\n")

    plt.title(f"{network_titles[i]} σ={sigma*eig:.3f}/λ", fontsize=14)
    plt.legend(fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("removed fraction, p", fontsize = 12)
    plt.ylabel("Largest Connected Component", fontsize = 12)
    plt.savefig(base + "lcc.png", dpi=300, bbox_inches='tight')
    print(f"Plot saved for {network_titles[i]}")
    plt.close()

