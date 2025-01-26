from network_model import ReliableNetwork
from genetic_algorithm import GeneticAlgorithm
import matplotlib.pyplot as plt
import networkx as nx
from data_loader import load_network_data
from resilience_evaluation import analyze_resilience

if __name__ == "__main__":
    filename = "network_topology_data.txt"
    num_nodes, edge_probabilities, edge_costs = load_network_data(filename) 

    network = ReliableNetwork(num_nodes, edge_probabilities)

    ga = GeneticAlgorithm(
        population_size=100,
        mutation_rate=0.1,
        num_generations=50,
        network=network,
        edge_probabilities=edge_probabilities
    )

    best_solution = ga.evolve()
    print("Melhor configuração de rede encontrada:", best_solution)

    # Carregar dados da rede
    num_nodes, edge_probabilities, edge_costs = load_network_data("network_topology_data.txt")

    # Criar grafo com NetworkX
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    for (u, v) in edge_probabilities.keys():
        G.add_edge(u, v, weight=edge_probabilities[(u, v)])

    # Desenhar a rede
    plt.figure(figsize=(6, 6))
    pos = nx.spring_layout(G)  # Layout do grafo
    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=1000)
    plt.show()

     # Criar o grafo otimizado no NetworkX
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    for (u, v), p in best_solution.items():
        if p > 0:  # Apenas adiciona arestas que ainda existem
            G.add_edge(u, v, weight=p)

    # Analisar a resiliência da rede otimizada
    print("\n=== Análise de Resiliência ===")
    resilience_results = analyze_resilience(G)

    # Exibir resultados finais
    print("\n=== Resultados da Resiliência ===")
    print(f"Conectividade mínima (arestas críticas): {resilience_results['min_cut_edges']}")
    print(f"Número médio de caminhos alternativos: {resilience_results['avg_paths']:.2f}")
    print(f"Nós críticos identificados: {resilience_results['central_nodes']}")
    print(f"Impacto da falha dos nós críticos: {resilience_results['critical_failures']}")  