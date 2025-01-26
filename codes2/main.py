import random
import csv
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class Coordenada:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class No:
    def __init__(self, coordenada):
        self.coordenada = coordenada

class Conexoes:
    def __init__(self):
        self.arestas = []  # Lista de tuplas representando as conexões entre os nós
        self.probabilidade_falhas_arestas = {}  # Dicionário com a probabilidade de falha de cada aresta

class Solucao:
    def __init__(self, nos):
        self.nos = nos  # Lista de nós (os nós já definidos)
        self.conexoes = Conexoes()  # Atributo conexões com arestas e probabilidade de falha
        self.fitness = 0  # Atributo fitness (confiabilidade da rede)

    def adicionar_aresta(self, aresta, prob_falha):

        self.conexoes.arestas.append(aresta)
        self.conexoes.probabilidade_falhas_arestas[aresta] = prob_falha
        self.fitness = self.calcular_fitness()  # Recalcula a fitness após adicionar a aresta

    def remover_arestas(self, arestas_remover):
        # Filtra as arestas que não devem ser removidas
        self.conexoes.arestas = [aresta for aresta in self.conexoes.arestas if aresta not in arestas_remover]

        # Também remove as probabilidades de falha associadas às arestas removidas
        for aresta in arestas_remover:
            if aresta in self.conexoes.probabilidade_falhas_arestas:
                del self.conexoes.probabilidade_falhas_arestas[aresta]

        self.fitness = self.calcular_fitness()  # Recalcula a fitness após a remoção das arestas

    def calcular_fitness(self, num_simulations = 1000):
        graph = nx.Graph()
        graph.add_edges_from(self.conexoes.arestas)

        connected_count = 0
    
        for _ in range(num_simulations):
            temp_graph = graph.copy()
            
            for edge in list(temp_graph.edges()):
                if np.random.rand() > self.conexoes.probabilidade_falhas_arestas[tuple(sorted(edge))]:
                    temp_graph.remove_edge(*edge)

            if nx.is_connected(temp_graph):
                connected_count += 1
        
        connectivity_probability = connected_count / num_simulations
        return connectivity_probability

    def is_connected(self):
        """ Verifica se a rede é conectada (usando DFS ou BFS)"""
        # Criar um grafo com as arestas
        graph = {i: [] for i in range(len(self.nos))}

        for (i, j) in self.conexoes.arestas:
            graph[i].append(j)
            graph[j].append(i)

        # Função DFS para verificar conectividade
        def dfs(node, visited):
            visited.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, visited)

        # Começar DFS de qualquer nó (usando o nó 0)
        visited = set()
        dfs(0, visited)

        # Se todos os nós foram visitados, a rede é conectada
        return len(visited) == len(self.nos)

    def find_critical_edges(self):
        """ Encontra as arestas críticas (pontes) que, se removidas, desconectam a rede """
        # Criar um grafo com as arestas
        graph = {i: [] for i in range(len(self.nos))}
        for (i, j) in self.conexoes.arestas:
            graph[i].append(j)
            graph[j].append(i)

        # Variáveis para DFS
        discovery_time = [-1] * len(self.nos)
        low = [-1] * len(self.nos)
        parent = [-1] * len(self.nos)
        bridges = []
        time = [0]  # Usando lista para modificar a variável dentro da função recursiva

        # Função DFS para encontrar as pontes
        def dfs(u):
            discovery_time[u] = low[u] = time[0]
            time[0] += 1

            for v in graph[u]:
                if discovery_time[v] == -1:  # Se v não foi visitado
                    parent[v] = u
                    dfs(v)

                    # Verificar se a aresta u-v é uma ponte
                    low[u] = min(low[u], low[v])
                    if low[v] > discovery_time[u]:
                        bridges.append((u, v))
                elif v != parent[u]:
                    low[u] = min(low[u], discovery_time[v])

        # Iniciar DFS de qualquer nó (usando o nó 0)
        for i in range(len(self.nos)):
            if discovery_time[i] == -1:
                dfs(i)

        return bridges

class Metaheuristica:
    def __init__(self):
        self.nos = []
        self.num_generations = 100  # Número de gerações
        self.population_size = 1  # Tamanho da população (número de redes)
        self.mutation_rate = 0.1  # Taxa de mutação
        self.solucoes = []  # Lista para armazenar as soluções geradas
        plt.ion()
        self.fig, self.ax = plt.subplots()

    def ler_csv(self, arquivo_csv):
        """ Lê os dados de um arquivo CSV e cria os objetos 'No' """
        with open(arquivo_csv, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Extrair os dados de cada linha
                x = float(row['x'])
                y = float(row['y'])

                # Criar os objetos Coordenada e No
                coordenada = Coordenada(x, y)
                no = No(coordenada)

                # Adicionar o nó à lista de nós
                self.nos.append(no)

        # Imprimir a lista de nós para verificar se os dados foram carregados corretamente
        self.imprimir_nos()

    def imprimir_nos(self):
        """ Imprime os dados dos nós """
        for i, no in enumerate(self.nos):
            print(f"No {i+1}: Coord: ({no.coordenada.x}, {no.coordenada.y})")

    def initialize_population(self):
        """ Inicializa a população com redes aleatórias """
        population = []
        for _ in range(self.population_size):
            solucao = self.generate_random_solution()
            population.append(solucao)
            self.solucoes.append(solucao)  # Armazenando a solução gerada
        return population

    def generate_random_solution(self):
        """ Gera uma solução aleatória (rede) """
        solucao = Solucao(self.nos)
        
        # Geração aleatória de arestas entre os nós
        for i in range(len(self.nos)):
            for j in range(i + 1, len(self.nos)):
                if random.random() > 0.5:  # 50% de chance de conexão entre os nós
                    prob_falha = random.uniform(0.5, 1)  # Probabilidade de falha aleatória entre 0 e 0.05
                    solucao.adicionar_aresta((i, j), prob_falha)  # Adiciona a aresta e sua probabilidade de falha à solução
        return solucao

    def plot_graph(self, solution, title="Solução"):
        """ Plota o gráfico da solução """
        G = nx.Graph()
        
        # Adicionando os nós ao grafo
        for i, no in enumerate(self.nos):
            G.add_node(i, pos=(no.coordenada.x, no.coordenada.y))

        # Adicionando as arestas ao grafo
        G.add_edges_from(solution.conexoes.arestas)

        # Extraindo as posições dos nós para plotagem
        pos = nx.get_node_attributes(G, 'pos')

        # Plotando o grafo
        plt.figure(figsize=(8, 6))
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, font_size=12, font_weight='bold', edge_color='gray')
        plt.title(title)
        plt.show()
    
    def select_parents(self, population, fitness_scores):
        selected = random.choices(population, weights=fitness_scores, k=2)
        return selected[0], selected[1]
    
    def crossover(self, parent1, parent2):
        return parent1 if np.random.rand() < 0.5 else parent2
    
    def mutate(self, individual, mutation_rate):
        if random.random() < mutation_rate:
            if len(individual.nos) > 0 and random.random() < 0.5:
                # Remove a random edge
                aresta_a_ser_removida = random.choice(list(individual.conexoes.arestas))
                individual.remover_arestas([aresta_a_ser_removida])
            else:
                # Adiciona uma aresta aleatória
                possible_edges = set(itertools.combinations(range(len(individual.nos)), 2)) - set(individual.conexoes.arestas)
                if possible_edges:
                    edge_to_add = random.choice(list(possible_edges))
                    prob_falha = random.uniform(0.5, 1)  # Probabilidade de falha aleatória entre 0 e 1
                    individual.adicionar_aresta(edge_to_add, prob_falha)
        return individual


    def evolve(self):
        """ Evolui a população usando crossover, mutação e seleção """
        population = self.initialize_population()
        best_individual = None
        best_fitness = 0
        
        for generation in range(self.num_generations):
            fitness_scores = [solucao.fitness for solucao in self.solucoes]
            new_population = []
            max_fitness = max(fitness_scores)
            if max_fitness > best_fitness:
                best_fitness = max_fitness
                best_individual = self.solucoes[np.argmax(fitness_scores)]
                print(best_individual)
                self.plot_graph(best_individual)
            
            print(f"Generation {generation+1}: Best Fitness = {best_fitness:.4f}")

            while len(new_population) < self.population_size:
                parent1, parent2 = self.select_parents(population, fitness_scores)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child, self.mutation_rate)
                new_population.append(child)

            population = new_population

        return best_individual, best_fitness


        # for generation in range(self.num_generations):
        #     population.sort(key=lambda x: x.fitness, reverse=True)  # Ordena pela fitness

        #     # Seleção dos melhores indivíduos para a próxima geração
        #     next_generation = population[:int(self.population_size / 2)]  # Seleciona metade

        #     # Crossover e mutação
        #     new_population = []
        #     while len(new_population) < self.population_size:
        #         parent1, parent2 = random.sample(next_generation, 2)
        #         child = self.crossover(parent1, parent2)
        #         child = self.mutate(child)
        #         new_population.append(child)

        #     population = new_population

        # Melhor solução
        best_solution = max(population, key=lambda x: x.fitness)
        return best_solution

    def plot_graph(self, individual):
        self.ax.clear()
        posicao_nos = [(no.coordenada.x, no.coordenada.y) for no in individual.nos]
        print(posicao_nos)
        # Plotando o grafo
        G = nx.Graph()
        G.add_edges_from(individual.conexoes.arestas)
        nx.draw(G, posicao_nos, with_labels=True, ax=ax, node_color='lightblue', edge_color='gray', node_size=700, font_weight='bold')
        plt.draw()
        plt.pause(0.1)

# Exemplo de uso
metaheuristica = Metaheuristica()
plt.ion()
fig, ax = plt.subplots()
# Chame a função para ler os dados do arquivo CSV
metaheuristica.ler_csv('nodes.csv')  # Substitua 'dados_nos.csv' pelo caminho do seu arquivo CSV

# Evolua a rede para encontrar a melhor topologia
best_network = metaheuristica.evolve()

# Imprima as soluções geradas para verificar
# print("\nSoluções geradas:")
# for i, solucao in enumerate(metaheuristica.solucoes):
#     print(f"Solução {i + 1}:")
#     print(f"Arestas: {solucao.conexoes.arestas}")
#     print(f"Probabilidade de falhas das arestas: {solucao.conexoes.probabilidade_falhas_arestas}")
#     print(f"Fitness (Confiabilidade): {solucao.fitness}")
#     print(f"Conectada? {solucao.is_connected()}")
#     print(f"Arestas críticas: {solucao.find_critical_edges()}")
