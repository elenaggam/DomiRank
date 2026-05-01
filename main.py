import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
import scipy
import scipy.sparse
import os
import time

import functions as f


G = nx.grid_2d_graph(7, 7, periodic=False) #create a grid graph
eigenvalues, _ = eigsh(nx.to_scipy_sparse_array(G).astype(float))
min = np.min(eigenvalues)
del eigenvalues, _

attack = f.generate_attack(f.domirank(G, sigma=-0.999/min)) #generate the attack using the centrality (descending)
lcc, links = f.network_attack_sampled(G, attack, sampling = 1) #attack the network and



out = f"Grid7/"
if not os.path.exists(out):
    os.makedirs(out)
x = np.linspace(0,1, lcc.shape[0])
plt.plot(x, lcc)
plt.xlabel('fraction of nodes removed')
plt.ylabel('largest connected component')
plt.savefig(out + f"0.999.png", dpi=300, bbox_inches='tight')

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