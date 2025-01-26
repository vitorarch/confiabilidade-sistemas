import numpy as np
import networkx as nx

def monte_carlo_reliability(num_simulations, network, edge_probabilities):
    """
    Estima a confiabilidade da rede usando Simulação de Monte Carlo.

    Parâmetros:
    - num_simulations: Número de simulações a serem executadas.
    - network: Objeto da classe ReliableNetwork.
    - edge_probabilities: Dicionário com as probabilidades de falha das arestas.

    Retorna:
    - Confiabilidade estimada da rede (probabilidade de ela permanecer conectada).
    """
    success_count = 0  # Contador de simulações onde a rede permanece conectada

    for _ in range(num_simulations):
        # Criar uma cópia da rede para simular falhas
        temp_graph = network.graph.copy()

        # Verificar a falha de cada aresta com base em sua probabilidade
        for (u, v) in list(temp_graph.edges()):
            # Garantir que acessamos a chave corretamente, independentemente da ordem
            edge_key = (u, v) if (u, v) in edge_probabilities else (v, u)
            
            if edge_key in edge_probabilities:  # Verifica se a aresta realmente existe no dicionário
                random_value = np.random.rand()  # Gerar valor aleatório entre 0 e 1
                failure_prob = edge_probabilities[edge_key]

                if random_value < failure_prob:  # Se o valor gerado for menor que a probabilidade, falha a aresta
                    temp_graph.remove_edge(u, v)

        # Testar se a rede continua conectada
        if nx.is_connected(temp_graph):
            success_count += 1  # Incrementar se a rede sobreviveu

    # Calcular a confiabilidade estimada (quantas vezes a rede sobreviveu / total de simulações)
    reliability = success_count / num_simulations
    print(reliability)
    return reliability
