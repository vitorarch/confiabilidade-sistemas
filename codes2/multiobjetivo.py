import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
import math

def pareto_dominates(sol1, sol2):
    fitness1 = fitness(sol1)
    fitness2 = fitness(sol2)

    return all(f1 >= f2 for f1, f2 in zip(fitness1, fitness2)) and any(f1 > f2 for f1, f2 in zip(fitness1, fitness2))

def pareto_dominates2(pareto_front):
    resposta = []
    
    for sol1 in pareto_front:
        # Verifica se a solução já é dominada por alguma solução existente na resposta
        dominated = False
        for sol2 in resposta:
            # Verifica se a sol2 domina sol1
            fitness1 = fitness(sol1)
            fitness2 = fitness(sol2)
            if all(f2 >= f1 for f1, f2 in zip(fitness1, fitness2)) and any(f2 > f1 for f1, f2 in zip(fitness1, fitness2)):
                dominated = True
                break
        
        if not dominated:
            resposta.append(sol1)
    
    return resposta



def fitness(individual):
    reliability = simulate_connectivity(individual, edge_reliabilities, num_simulations)
    cost = calculate_cost(individual)
    return reliability, -cost

def calculate_cost(individual):
    total_cost = 0
    for edge in individual.edges():
        node1, node2 = edge
        x1, y1 = node_positions[node1]
        x2, y2 = node_positions[node2]
        total_cost += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return total_cost

def simulate_connectivity(graph, edge_reliabilities, num_simulations=10000):
    connected_count = 0
    
    for _ in range(num_simulations):
        temp_graph = graph.copy()
        for edge in list(temp_graph.edges()):
            if np.random.rand() > edge_reliabilities[edge]:
                temp_graph.remove_edge(*edge)
        
        if nx.is_connected(temp_graph):
            connected_count += 1
    
    return connected_count / num_simulations

def generate_random_graph(num_nodes, max_edges):
    G = nx.Graph()
    
    G.add_nodes_from(range(num_nodes))
    
    possible_edges = [(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes)]
    selected_edges = random.sample(possible_edges, min(max_edges, len(possible_edges)))
    
    G.add_edges_from(selected_edges)
    
    return G

def initialize_population():
    return [generate_random_graph(28, 40) for _ in range(population_size)]

def select_parents(population, pareto_front):
    selected = random.choices(pareto_front, k=2)
    return selected[0], selected[1]

def crossover(parent1, parent2):
    children = nx.Graph()
    # Combina as arestas dos dois pais
    edges1 = list(parent1.edges())
    edges2 = list(parent2.edges())
    max_edges = max(len(edges1), len(edges2))
    
    for i in range(max_edges):
        if random.random() < 0.5:
            if i < len(edges1):
                children.add_edge(*edges1[i])
        else:
            if i < len(edges2):
                children.add_edge(*edges2[i])
                
    return children

def mutate(individual):
    if random.random() < mutation_rate:
        possible_edges = set(nx.complete_graph(individual.nodes()).edges()) - set(individual.edges())
        if possible_edges:
            edge_to_add = random.choice(list(possible_edges))
            individual.add_edge(*edge_to_add)
    return individual

def genetic_algorithm(edge_reliabilities, population_size, generations, mutation_rate, num_simulations):
    # Inicializar população
    population = initialize_population()
    pareto_front = []

    for generation in range(generations):
        new_population = []
        fitness_scores = [fitness(individual) for individual in population]

        # Iteração sobre a população e impressão dos valores de fitness de cada indivíduo
        for i, individual in enumerate(population):
            reliability, cost = fitness(individual)  # Obter os valores de fitness (confiabilidade e custo)
            print(f"Indivíduo {i+1}: Confiabilidade = {reliability}, Custo = {-cost}")

        # Selecionar as soluções não dominadas (pareto front)
        for individual in population:
            # Verifica se a solução é conectada e não dominada
            if nx.is_connected(individual) and not any(pareto_dominates(individual, other) for other in pareto_front):
                pareto_front.append(individual)

        print(f"Generation {generation+1}: Pareto Front size = {len(pareto_front)}")

        # Gerar nova população
        while len(new_population) < population_size:
            parent1, parent2 = select_parents(population, pareto_front)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

    #result = (pareto_front)
    return pareto_front

def plot_graph(individual):
    ax.clear()

    # Atribuindo as posições dos nós diretamente do node_positions
    pos = {i: node_positions[i] for i in individual.nodes()}

    # Plotando o grafo
    nx.draw(individual, pos, with_labels=True, ax=ax, node_color='lightblue', edge_color='gray', node_size=700, font_weight='bold')
    plt.draw()
    plt.pause(0.1)

def plot_pareto_front(pareto_front):
    # Obter os objetivos (confiabilidade e custo) de cada solução
    reliability_values = []
    cost_values = []
    
    for solution in pareto_front:
        reliability, cost = fitness(solution)  # Obter os valores de confiabilidade e custo
        reliability_values.append(reliability)
        cost_values.append(cost)
    
    # Plotar os valores de confiabilidade vs custo
    plt.scatter(reliability_values, cost_values, color='blue', label='Soluções de Pareto')
    
    # Configurar os eixos
    plt.xlabel('Confiabilidade')
    plt.ylabel('Custo')
    plt.title('Fronteira de Pareto')
    plt.grid(True)
    plt.legend()
    plt.show()

# Configurações e execução
plt.ion()
fig, ax = plt.subplots()

node_positions = [
    (-8.05389, -34.88111), (-9.66583, -35.73528), (-7.23056, -35.88111), (-7.115, -34.86306),
    (-15.77972, -47.92972), (-19.92083, -43.93778), (-10.91111, -37.07167), (-12.97111, -38.51083),
    (-20.31944, -40.33778), (-22.90278, -43.2075), (-3.10194, -60.025), (-16.67861, -49.25389),
    (-15.59611, -56.09667), (-25.42778, -49.27306), (-30.03306, -51.23), (-27.59667, -48.54917),
    (-23.5475, -46.63611), (-8.76194, -63.90389), (-9.97472, -67.81), (-10.21278, -48.36028),
    (-20.44278, -54.64639), (-5.08917, -42.80194), (-5.795, -35.20944), (2.81972, -60.67333),
    (0.03889, -51.06639), (-1.45583, -48.50444), (-2.52972, -44.30278), (-3.71722, -38.54306)
]


edge_reliabilities = {
    (i, j): np.random.uniform(0.9, 0.95) for i in range(28) for j in range(28)
}

population_size = 10
generations = 5
mutation_rate = 0.1
num_simulations = 500

# Rodar o algoritmo genético
pareto_front = genetic_algorithm(edge_reliabilities, population_size, generations, mutation_rate, num_simulations)

# Exibir as soluções não dominadas (fronteira de Pareto)
print("\nPareto Front Solutions:")
for solution in pareto_front:
    print(f"Solution: {solution}")
    plot_graph(solution)

plt.ioff()
plt.show()


for i, individual in enumerate(pareto_front):
    reliability, cost = fitness(individual)  # Obter os valores de fitness (confiabilidade e custo)
    print(f"Indivíduo {i+1}: Confiabilidade = {reliability}, Custo = {-cost}")

plot_pareto_front(pareto_front)