import networkx as nx
import math as m
import numpy as np
import matplotlib.pyplot as plt

def calculate_graph_reliability(graph, num_simulations=1000):
    connected_count = 0
    
    for _ in range(num_simulations):
        temp_graph = graph.copy()
        
        for edge in list(temp_graph.edges()):
            if np.random.rand() > edge_reliabilities[edge]:
                temp_graph.remove_edge(*edge)

        if nx.is_connected(temp_graph):
            connected_count += 1
    
    connectivity_probability = connected_count / num_simulations
    return connectivity_probability

nos = [(0, 0), (1, 1), (0, 1)]
arestas = [(0, 1), (0, 2), (1, 2)]

G = nx.Graph()
        
for i, no in enumerate(nos):
    G.add_node(i, pos=(no[0], no[1]))

G.add_edges_from(arestas)

pos = nx.get_node_attributes(G, 'pos')

plt.figure(figsize=(8, 6))
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, font_size=12, font_weight='bold', edge_color='gray')
plt.show()


edge_reliabilities = {
    (0, 1): 0.9,
    (0, 2): 0.8,
    (1, 2): 0.9
}

reliability = calculate_graph_reliability(G, 1000)
print(f"Reliability of the graph: {reliability:.4f}")


