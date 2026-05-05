import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
import scipy
import scipy.sparse
import os
import time

import functions as f

out = "ER/attack/"
if not os.path.exists(out):
    os.makedirs(out)

sigma = 0.9
plotting = [0.0, 0.09, 0.19]

G = nx.erdos_renyi_graph(32, 0.12, seed=82) #create an ER graph
G = f.relabel_nodes(G) #relabel the nodes to be from 0 to N-1 instead of tuples
eigenvalues, _ = eigsh(nx.to_scipy_sparse_array(G).astype(float))
min = np.min(eigenvalues)
del eigenvalues, _

psi = f.domirank(G, sigma=-sigma/min)
between = list(nx.betweenness_centrality(G).values()) #compute the centrality measures
close = list(nx.closeness_centrality(G).values())
page = list(nx.pagerank(G, alpha=-sigma/min, max_iter=1000).values()) #compute the centrality measures

attack = f.generate_attack(psi) #generate the attack using the centrality (descending)
attack_betweenness = f.generate_attack(between) #generate the attack using the centrality (descending)
attack_closeness = f.generate_attack(close) #generate the attack using the centrality (
attack_pagerank = f.generate_attack(page) #generate the attack using the centrality (descending)

lcc, links = f.network_attack_sampled(nx.to_scipy_sparse_array(G), attack, sampling =1) #attack the network and
lcc_betweenness, links_betweenness = f.network_attack_sampled(nx.to_scipy_sparse_array(G), attack_betweenness, sampling =1) #attack the network and
lcc_closeness, links_closeness = f.network_attack_sampled(nx.to_scipy_sparse_array(G), attack_closeness, sampling =1) #attack the network and
lcc_pagerank, links_pagerank = f.network_attack_sampled(nx.to_scipy_sparse_array(G), attack_pagerank, sampling =1) #attack the network and

f.network_attack_plotting(G, attack, plotting = plotting, psi = psi, directory = out + f"domirank_{sigma}_") #attack the network and
f.network_attack_plotting(G, attack_betweenness, plotting = plotting, psi = between, directory = out + f"betweenness_{sigma}_") #attack the network and
f.network_attack_plotting(G, attack_closeness, plotting = plotting, psi = close, directory = out + f"closeness_{sigma}_") #attack the network and
f.network_attack_plotting(G, attack_pagerank, plotting = plotting, psi = page, directory = out + f"pagerank_{sigma}_") #attack the network and

x = np.linspace(0,1, lcc.shape[0])
plt.plot(x, lcc, label = 'domirank')
plt.plot(x, lcc_betweenness, label = 'betweenness')
plt.plot(x, lcc_closeness, label = 'closeness')
plt.plot(x, lcc_pagerank, label = 'pagerank')
plt.axvline(x=plotting[1], color='grey', linestyle='--')
plt.axvline(x=plotting[2], color='grey', linestyle='--')
plt.legend(fontsize = 14)
plt.xlabel('fraction of nodes removed')
plt.ylabel('largest connected component')
plt.savefig(out + f"lcc_{sigma}.png", dpi=300, bbox_inches='tight')
plt.close()

plt.plot(x, links, label = 'domirank')
plt.plot(x, links_betweenness, label = 'betweenness')
plt.plot(x, links_closeness, label = 'closeness')
plt.plot(x, links_pagerank, label = 'pagerank')
plt.axvline(x=plotting[1], color='grey', linestyle='--')
plt.axvline(x=plotting[2], color='grey', linestyle='--')
plt.legend(fontsize = 14)
plt.xlabel('fraction of nodes removed')
plt.ylabel('number of links')
plt.savefig(out + f"links_{sigma}.png", dpi=300, bbox_inches='tight')
plt.close()

# time_file = open(out + "times.txt", "w")
# fig, axes = plt.subplots(2, 2, figsize=(13,12))
# ax = axes.flatten()
# i=0
# for sigma in [ 0.01, 0.75, 0.95, 0.999]:
    
#     start_time = time.time()
#     gamma = f.domirank(G, sigma=-sigma/min)
#     domirank_time = time.time() - start_time
#     time_file.write(f'{domirank_time:.6f}\n')

#     pos = nx.spring_layout(G, seed=200)
#     nx.draw(G, pos, cmap=plt.get_cmap('cividis'), node_color=gamma, font_color='white', ax=ax[i])
#     ax[i].set_title(r"$\sigma = -$" + f"{-sigma:.2f}/"+r"$\lambda_N$", fontsize = 18)

#     i+=1
# plt.savefig(out + f"paper.png", dpi=300, bbox_inches='tight')
# plt.close()
# time_file.close()