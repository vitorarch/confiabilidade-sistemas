import random
import numpy as np
from reliability_evaluation import monte_carlo_reliability

class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, num_generations, network, edge_probabilities):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.num_generations = num_generations
        self.network = network
        self.edge_probabilities = edge_probabilities
        self.population = self.initialize_population()

    def initialize_population(self):
        """
        Cria uma população inicial aleatória de configurações de redes.
        """
        return [self.random_solution() for _ in range(self.population_size)]

    def random_solution(self):
        """
        Gera uma solução aleatória removendo algumas arestas da rede.
        """
        solution = self.edge_probabilities.copy()
        for edge in solution.keys():
            if random.random() < 0.2:  # Probabilidade de remoção de aresta inicial
                solution[edge] = 1.0  # Aresta sempre falha (removida)
        return solution

    def fitness(self, solution):
        """
        Avalia a confiabilidade de uma solução de rede modificada.
        """
        self.network.reset_network(solution)
        return monte_carlo_reliability(1000, self.network, solution)  # 1000 simulações

    def crossover(self, parent1, parent2):
        """
        Realiza o crossover entre dois pais gerando um novo filho.
        """
        child = {}
        for edge in parent1.keys():
            child[edge] = parent1[edge] if random.random() < 0.5 else parent2[edge]
        return child

    def mutate(self, solution):
        """
        Aplica mutação a uma solução.
        """
        for edge in solution.keys():
            if random.random() < self.mutation_rate:
                solution[edge] = max(0, min(1, solution[edge] + random.uniform(-0.1, 0.1)))
        return solution

    def evolve(self):
        """
        Executa o Algoritmo Genético.
        """
        for _ in range(self.num_generations):
            self.population.sort(key=self.fitness, reverse=True)
            new_population = self.population[:10]  # Mantém os 10 melhores indivíduos
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(self.population[:50], 2)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            self.population = new_population
        
        return self.population[0]  # Melhor solução final
