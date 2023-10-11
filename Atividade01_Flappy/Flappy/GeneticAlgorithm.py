from MLP import MLP
from FlappyBird import FlappyBird_GeneticMLP
import random
import copy
import numpy as np


class GeneticAlgorithm:
    def __init__(self, entrada, saida, populationSize=100, mutation_rate=0.05, generations=100, early_stop=10):
        self.entrada = entrada
        self.saida = saida
        self.populationSize = populationSize
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.early_stop = early_stop

        self.population = []
        self.best_individual = None
        self.best_fitness = float('-inf')

    def generateInitialPopulation(self):
        for _ in range(self.populationSize):
            # Create a random MLP architecture
            # You can adjust the range based on your needs
            hidden_neurons = random.randint(1, 20)
            mlp = MLP(self.entrada, hidden_neurons,
                      self.saida, taxaDeAprendizado=0.1)
            self.population.append(mlp)
    
    def fitnessFunction(self, mlp):
        game = FlappyBird_GeneticMLP(mlp) 
        # game.jump()
        while not game.dead:
            inputs = game.getInputs()
            print(inputs)
            output = mlp.feedForward(np.array(inputs).reshape(-1, 1))[1]
            print(output)
            jump_threshold = 0.5
            should_jump = output[0][0] > jump_threshold
            game.birdUpdate(should_jump)
            print(game.score)

        return game.score

    def evaluateIndividual(self, mlp):
        fitness = self.fitnessFunction(mlp)
        return fitness

    def selectParents(self):
        tournament_size = 3
        tournament = random.sample(self.population, tournament_size)
        parent = max(tournament, key=lambda mlp: self.evaluateIndividual(mlp))
        return parent

    def generateChildren(self, parent1, parent2):
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)

        # Crossover simples: trocar os pesos dos pais para criar os filhos
        child1.weights_input_hidden = parent2.weights_input_hidden
        child2.weights_input_hidden = parent1.weights_input_hidden
        print(child1.weights_input_hidden)
        print(child2.weights_input_hidden)
        return child1, child2

    def mutate(self, mlp):
        if random.uniform(0, 1) < self.mutation_rate:
            if random.choice([True, False]):
                new_neuron = [random.uniform(-1, 1) for _ in range(len(mlp.weights_input_hidden[0]))]
                mlp.weights_input_hidden.append(new_neuron)
            else:
                if len(mlp.weights_input_hidden) > 1:
                    idx = random.randint(0, len(mlp.weights_input_hidden) - 1)
                    del mlp.weights_input_hidden[idx]
        return mlp

    def execute(self):
        self.generateInitialPopulation()

        for generation in range(self.generations):
            new_population = []
            for _ in range(self.populationSize // 2):
                parent1 = self.selectParents()
                parent2 = self.selectParents()
                child1, child2 = self.generateChildren(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])

            self.population = new_population

        # Avalia o fitness de cada indivíduo na população
            best_in_generation = None
            best_fitness_in_generation = float('-inf')
            for mlp in self.population:
                fitness = self.evaluateIndividual(mlp)
                if fitness > best_fitness_in_generation:
                    best_fitness_in_generation = fitness
                    best_in_generation = mlp

        # Verifica se o critério de parada antecipada é atendido
            if generation > self.early_stop:
                if all(mlp == best_in_generation for mlp in self.population[-self.early_stop:]):
                    break  

        # Atualiza o melhor indivíduo global
        if best_fitness_in_generation > self.best_fitness:
            self.best_fitness = best_fitness_in_generation
            self.best_individual = best_in_generation
            
        return self.best_individual