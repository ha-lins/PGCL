import numpy as np
import dgl
import torch as th
import networkx as nx

node = 118


aa_data = np.load('data/aa/aa_data.npy', allow_pickle=True)
aa_label = np.load('data/aa/aa_label.npy', allow_pickle=True)

# construct networkx graphs
graphs = []
labels = []
for trial in range(0, 280):
    label = aa_label[trial]
    trial_graphs = []
    for graph in range(0, 10):
        g = nx.Graph()
        for i in range(0, node):
            for j in range(0, node):
                g.add_edge(i, j, weight=aa_data[trial][graph][i][j])
        u, v = th.tensor(np.array(g.edges)[:, 0]), np.array(g.edges)[:, 1]
        weights = th.zeros(g.number_of_edges())
        for edge in range(0, g.number_of_edges()):
            np_u = np.array(g.edges)[:, 0][edge]
            np_v = np.array(g.edges)[:, 1][edge]
            weights[edge] = th.tensor(g.edges[np_u, np_v]['weight'])

        # construct dgl graph
        G = dgl.graph((u, v))
        G.edata['w'] = weights
        print(G.num_edges())
        exit()

        trial_graphs.append(G)
    graphs.append(trial_graphs)



