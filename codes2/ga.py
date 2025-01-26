import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt

# Simulate connectivity function (same as earlier)
def fitness(graph, edge_reliabilities, num_simulations=10000):
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

# Genetic Algorithm
def genetic_algorithm(edge_reliabilities, population_size, generations, mutation_rate, num_simulations):
    # Initialize population
    def initialize_population():
        return [
            generate_random_graph(10, 12)
            for _ in range(population_size)
        ]

    # Fitness function
    def fitness(individual):
        return fitness(individual, edge_reliabilities, num_simulations)

    # Selection
    def select_parents(population, fitness_scores):
        selected = random.choices(population, weights=fitness_scores, k=2)
        return selected[0], selected[1]

    # Crossover
    def crossover(parent1, parent2):
        return parent1 if np.random.rand() < 0.5 else parent2

    # Mutation
    def mutate(individual):
        if random.random() < mutation_rate:
            if len(individual.edges()) > 0 and random.random() < 0.5:
                # Remove a random edge
                edge_to_remove = random.choice(list(individual.edges()))
                individual.remove_edge(*edge_to_remove)
            else:
                # Add a random edge
                possible_edges = set(nx.complete_graph(individual.nodes()).edges()) - set(individual.edges())
                if possible_edges:
                    edge_to_add = random.choice(list(possible_edges))
                    individual.add_edge(*edge_to_add)
        return individual

    # Genetic Algorithm Main Loop
    population = initialize_population()
    best_individual = None
    best_fitness = 0

    for generation in range(generations):
        fitness_scores = [fitness(individual) for individual in population]
        new_population = []

        # Track the best individual
        max_fitness = max(fitness_scores)
        if max_fitness > best_fitness:
            best_fitness = max_fitness
            best_individual = population[np.argmax(fitness_scores)]
            print(best_individual)
            plot_graph(best_individual)

        print(f"Generation {generation+1}: Best Fitness = {best_fitness:.4f}")

        # Generate new population
        while len(new_population) < population_size:
            parent1, parent2 = select_parents(population, fitness_scores)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

    return best_individual, best_fitness

def plot_graph(individual):
    print(individual.edges())
    ax.clear()
    # Extraindo as posições dos nós para plotagem
    pos = nx.get_node_attributes(individual, 'pos')

    # Plotando o grafo
    nx.draw(individual, node_positions, with_labels=True, ax=ax, node_color='lightblue', edge_color='gray', node_size=700, font_weight='bold')
    plt.draw()
    plt.pause(0.1)

plt.ion()
fig, ax = plt.subplots()

node_positions = [
    (10.0, 0.0),
    (8.09, 5.88),
    (3.09, 9.51),
    (-3.09, 9.51),
    (-8.09, 5.88),
    (-10.0, 0.0),
    (-8.09, -5.88),
    (-3.09, -9.51),
    (3.09, -9.51),
    (8.09, -5.88)
]

edge_reliabilities = {
    (i, j): np.random.uniform(0.8, 0.95) for i in range(10) for j in range(10)
}

population_size = 50
generations = 50
mutation_rate = 0.1
num_simulations = 500

# Run the Genetic Algorithm
best_solution, best_solution_fitness = genetic_algorithm(edge_reliabilities, population_size, generations, mutation_rate, num_simulations)
print("\nBest Solution Edge Reliabilities:")
print(best_solution)
print(f"Best Solution Fitness: {best_solution_fitness:.4f}")

plt.ioff()
plt.show()