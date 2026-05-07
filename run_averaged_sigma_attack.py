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