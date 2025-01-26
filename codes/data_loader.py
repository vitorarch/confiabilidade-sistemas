def load_network_data(filename):
    """
    Lê os dados da rede a partir de um arquivo e retorna os nós, arestas e suas propriedades.
    :param filename: Nome do arquivo contendo os dados
    :return: Número de nós, dicionário de arestas com probabilidades e custos
    """
    with open(filename, "r") as file:
        lines = file.readlines()
    
    num_nodes, num_edges = map(int, lines[0].split())
    edge_probabilities = {}
    edge_costs = {}

    for line in lines[1:num_edges+1]:
        u, v, failure_prob, cost = line.split()
        edge_probabilities[(min(u, v), max(u, v))] = float(failure_prob)  # Ordena os nós para evitar erros
        edge_costs[(int(u), int(v))] = int(cost)

    return num_nodes, edge_probabilities, edge_costs



