import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import functions as f
from scipy.sparse.linalg import eigsh


titles = ['Domirank', 'Betweenness', 'Closeness', 'PageRank']
# ER 0.12, seed=23
# BA 1, seed=82

# graph 
N = 25
avg_sigma = 0
avgN = 10
avg_eig = 0
file = open(f"BA_avg{avgN}.txt", "w")
for i in range(avgN):
    G = nx.barabasi_albert_graph(N, 1) # create the graph
    sparse_G = nx.to_scipy_sparse_array(G)
    G = f.relabel_nodes(G) # relabel the nodes to be from 0 to N-1 instead of tuples
    eigenvalues, _ = eigsh(nx.to_scipy_sparse_array(G).astype(float))
    eig = np.min(eigenvalues) # lambda_N

    # optimal sigma
    sigma, _ = f.old_optimal_sigma(sparse_G, endVal = eig, sampling = 1, iterationNo=1000) # compute the optimal sigma for the graph
    file.write(f"{sigma:.4f}\t{-sigma*eig:.4f}\n")
    avg_sigma += sigma
    avg_eig += eig

avg_sigma /= avgN
avg_eig /= avgN
file.close()
G = nx.barabasi_albert_graph(N, 1, seed=82) # create the graph
sparse_G = nx.to_scipy_sparse_array(G)
G = f.relabel_nodes(G) # relabel the nodes to be from 0 to N-1 instead of tuples

# data output
out = f"BA/attack_avg{avgN}_{-avg_sigma*avg_eig:.3f}/" # sigma times the minimum eigenvalue
if not os.path.exists(out):
    os.makedirs(out)
plotting = [0.0, 0.08, 0.18, 0.28, 0.4] # p of the attack to plot


# fig1 = plt.figure(1)
# ourRange = np.linspace(0,1, _.shape[0]) 
# index = np.where(_ == _.min())[0][-1]
# plt.plot(ourRange, _)
# plt.plot(ourRange[index],_[index], 'ro', mfc = 'none', markersize = 10)
# plt.xlabel('sigma')
# plt.ylabel('loss')
# plt.savefig(out+f"optimal_sigma_{-avg_sigma*eig:.2f}.png", dpi=300, bbox_inches='tight')
# plt.close()



# compute the centrality measures and attacks
psi = f.domirank(G, sigma=sigma) #compute the centrality measures
between = list(nx.betweenness_centrality(G).values()) #compute the centrality measures
close = list(nx.closeness_centrality(G).values())
page = list(nx.pagerank(G, max_iter=1000).values()) #compute the centrality measures

centralities = np.array([psi, between, close, page])

attack_domi = f.generate_attack(psi) #generate the attack using the centrality (descending)
attack_betweenness = f.generate_attack(between) #generate the attack using the centrality (descending)
attack_closeness = f.generate_attack(close) #generate the attack using the centrality (
attack_pagerank = f.generate_attack(page) #generate the attack using the centrality (descending)

attack = np.array([attack_domi, attack_betweenness, attack_closeness, attack_pagerank])


# plotting the attacks for different fractions of removed nodes
f.network_attack_plotting(G, attack, plotting, centralities, titles, directory = out) 


# plotting the largest connected component and number of links for the whole attack
lcc_domi, links_domi = f.network_attack_sampled(nx.to_scipy_sparse_array(G), attack_domi, sampling =1) #attack the network and
lcc_betweenness, links_betweenness = f.network_attack_sampled(nx.to_scipy_sparse_array(G), attack_betweenness, sampling =1) #attack the network and
lcc_closeness, links_closeness = f.network_attack_sampled(nx.to_scipy_sparse_array(G), attack_closeness, sampling =1) #attack the network and
lcc_pagerank, links_pagerank = f.network_attack_sampled(nx.to_scipy_sparse_array(G), attack_pagerank, sampling =1) #attack the network and

lcc = [lcc_domi, lcc_betweenness, lcc_closeness, lcc_pagerank]
links = [links_domi, links_betweenness, links_closeness, links_pagerank]
x = np.linspace(0,1, lcc_domi.shape[0])

for i in range(1, len(lcc)):
    plt.plot(x, lcc[i], label=titles[i], linewidth=2)
plt.plot(x, lcc[0], label=titles[0], linewidth=3, linestyle='dotted', color = 'b')
for p in plotting[1:]:
    plt.axvline(x=p, color='grey', linestyle='--', zorder=0)
plt.legend(fontsize=12)
plt.xlabel("removed fraction, p", fontsize = 12)
plt.ylabel("Largest Connected Component", fontsize = 12)
plt.savefig(out + "lcc.png", dpi=300, bbox_inches='tight')
plt.close()

for i in range(1, len(lcc)):
    plt.plot(x, links[i], label=titles[i], linewidth=2)
plt.plot(x, links[0], label=titles[0], linewidth=3, linestyle='dotted', color = 'b')
for p in plotting[1:]:
    plt.axvline(x=p, color='grey', linestyle='--', zorder=0)
plt.legend(fontsize=12)
plt.xlabel("removed fraction, p", fontsize = 12)
plt.ylabel("Number of Links", fontsize = 12)
plt.savefig(out + "links.png", dpi=300, bbox_inches='tight')
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