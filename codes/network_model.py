import networkx as nx
import random

class ReliableNetwork:
    def __init__(self, num_nodes, edge_probabilities):
        """
        Inicializa a rede com nós e arestas probabilísticas.
        :param num_nodes: Número de nós na rede
        :param edge_probabilities: Dicionário {(nó1, nó2): probabilidade_de_falha}
        """
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(num_nodes))
        for (u, v), p in edge_probabilities.items():
            self.graph.add_edge(u, v, failure_prob=p)

    def remove_failed_edges(self):
        """
        Simula falhas removendo arestas com base nas probabilidades de falha.
        """
        for u, v in list(self.graph.edges()):
            if random.random() < self.graph[u][v]['failure_prob']:
                self.graph.remove_edge(u, v)

    def is_connected(self):
        """
        Verifica se a rede permanece conectada após falhas.
        """
        return nx.is_connected(self.graph)

    def reset_network(self, edge_probabilities):
        """
        Restaura a rede original após simulação.
        """
        self.graph = nx.Graph()
        for (u, v), p in edge_probabilities.items():
            self.graph.add_edge(u, v, failure_prob=p)
