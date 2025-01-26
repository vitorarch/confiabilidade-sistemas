import networkx as nx

def analyze_resilience(G):
    """
    Analisa a resiliência da rede:
    - Calcula a conectividade mínima (k-conectividade)
    - Mede o número de caminhos alternativos
    - Identifica nós críticos
    - Verifica impacto da falha dos nós críticos
    """
    results = {}

    # 1. Conectividade mínima (quantas arestas precisam falhar para desconectar a rede)
    min_cut_edges = nx.edge_connectivity(G)
    results["min_cut_edges"] = min_cut_edges
    print(f"A conectividade mínima da rede é {min_cut_edges} (quantas arestas precisam falhar para desconectar a rede).")

    # 2. Número médio de caminhos alternativos entre pares de nós
    path_counts = []
    for u in G.nodes():
        for v in G.nodes():
            if u != v:
                paths = list(nx.all_simple_paths(G, source=u, target=v))
                path_counts.append(len(paths))

    avg_paths = sum(path_counts) / len(path_counts) if path_counts else 0
    results["avg_paths"] = avg_paths
    print(f"Número médio de caminhos alternativos: {avg_paths:.2f}")

    # 3. Identificação de nós críticos (nós com maior grau de conectividade)
    central_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)
    results["central_nodes"] = central_nodes[:3]
    print("Os nós mais críticos (com maior grau de conectividade) são:", central_nodes[:3])

    # 4. Impacto da falha dos nós críticos
    critical_failures = {}
    for node, _ in central_nodes[:3]:  # Verificar os 3 nós mais conectados
        G_copy = G.copy()
        G_copy.remove_node(node)
        is_connected = nx.is_connected(G_copy)
        critical_failures[node] = is_connected
        print(f"Se o nó {node} falhar, a rede permanece conectada? {is_connected}")

    results["critical_failures"] = critical_failures

    return results
