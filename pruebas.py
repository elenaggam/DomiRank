import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import functions as f
from scipy.sparse.linalg import eigsh

N = 1001
avgN = 10
 # 'ER (high degree)', 'ER (low degree)', 'BA',   nx.erdos_renyi_graph, nx.erdos_renyi_graph, nx.barabasi_albert_graph,   dict(n=N, p=20.0/(N-1.0)),  dict(n=N, p=6.0/(N-1.0)), dict(n=N, m=3), 
network_titles = ['WS',  'RGG', 'connected WS']
network_functions = [nx.watts_strogatz_graph, nx.random_geometric_graph, nx.connected_watts_strogatz_graph]
network_args = [dict(n=N, k=4, p=0.12), dict(n=N, radius=np.sqrt(16/(np.pi*N))), dict(n=N, k=4, p=0.12)]


startval =  0.000001
iterationNo = 100

for i in range(len(network_functions)):
    if not os.path.exists(f"test/{network_titles[i]}"):
        os.makedirs(f"test/{network_titles[i]}")

    avg_best_sigma = 0
    area_avg = []
    tempRange = []
    avg_eig = []

    figure = plt.figure()
    for a in range(avgN):   
        G = network_functions[i](**network_args[i])
        G = f.relabel_nodes(G) # relabel the nodes to be from 0
        sparse_G = nx.to_scipy_sparse_array(G)
        eigenvalues, _ = eigsh(nx.to_scipy_sparse_array(G).astype(float))
        eigval = np.min(eigenvalues) # lambda_N

        if nx.is_connected(G) == False:
            print(f"Warning: {network_titles[i]} is not connected, optimal sigma might not be accurate")

        
        best_sigma, areas_lcc = f.old_optimal_sigma(sparse_G, endVal = eigval, iterationNo = iterationNo)
        avg_best_sigma += best_sigma
        area_avg.append(areas_lcc)
        avg_eig.append(eigval)
        tempRange.append(np.arange(startval, -0.9999/eigval + (-0.9999/eigval-startval)/iterationNo, (-0.9999/eigval-startval)/iterationNo))

    size_lim = min(len(tempRange[a]) for a in range(avgN))
    tempRange = [tempRange[a][:size_lim] for a in range(avgN)]
    area_avg = [area_avg[a][:size_lim] for a in range(avgN)]
    for a in range(avgN):
        plt.plot(-tempRange[a]*avg_eig[a], area_avg[a], alpha = 0.5)

    total_avg_eig = np.mean(avg_eig)
    avg_best_sigma /= avgN
    avg_tempRange = -np.mean(tempRange, axis=0)*total_avg_eig
    area_avg = np.mean(area_avg, axis=0)
    
    plt.plot(avg_tempRange, area_avg, label = f"Average", color = "black", linewidth=3, linestyle='dotted')
    plt.axvline(x=-avg_best_sigma*total_avg_eig, color='red', linestyle='--', label=f'σ: {avg_best_sigma*total_avg_eig:.4f}/λ')
    best_sigma_2 = np.where(area_avg == np.min(area_avg))[0][-1]
    plt.axvline(x=avg_tempRange[best_sigma_2], color='blue', linestyle='--', label=f'σ: {avg_tempRange[best_sigma_2]:.4f}/λ')
    plt.legend()
    plt.xlabel("-σ/λ")
    plt.ylabel("Area under LCC curve")
    plt.title(f"Optimal σ for {network_titles[i]} (N={N})")
    plt.savefig(f"test/{network_titles[i]}/{N}_{avgN}_optimal_sigma.png", dpi = 300, bbox_inches='tight')
    plt.close()