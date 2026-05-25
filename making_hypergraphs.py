import hypernetx as hnx
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


np_array = np.array([['A','1'],['A','2'],['A','3'],['B','1'],['B','4'],['C','3'],['C','4']])

H = hnx.Hypergraph(np_array)

plt.subplots()
hnx.draw(H)
plt.show()