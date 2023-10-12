import pygad
import numpy as np
import random
import FlappyMLP

# GENOME =  [n_input, i0, i1, i2, i3, i4, i5, n_hidden, learn_rate, norm, bias]


class FlappyBirdGA:
    def __init__(self, num_generations=99, population_size=50, num_parents=5):
        self.GENOME = [
            "n_input",
            "i0",
            "i1",
            "i2",
            "i3",
            "i4",
            "i5",
            "n_hidden",
            "learn_rate",
            "norm",
        ]

        self.num_generations = num_generations
        self.population_size = population_size
        self.num_parents = num_parents

        initial_population = self.generate_initial_population(population_size)

        self.ga_instance = pygad.GA(
            num_generations=self.num_generations,
            num_parents_mating=self.num_parents,
            fitness_func=self.fitness_function,
            sol_per_pop=self.population_size,
            num_genes=len(self.GENOME),
            initial_population=initial_population,
            crossover_type="uniform",
            mutation_type="swap",
            parent_selection_type="tournament",
            on_mutation=self.ensure_valid_genomes,
        )

    def generate_initial_population(self, pop_size):
        """
        Generates an initial population of genomes.

        Args:
        pop_size (int): The number of genomes to generate.

        Returns:
        np.array: The initial population of genomes.
        """
        # Generate the initial population using the generate_genome function
        initial_population = [self.generate_genome() for _ in range(pop_size)]

        # Convert to a NumPy array and return
        return np.array(initial_population)

    def generate_genome(self):
        """
        Generates a genome with random values adhering to specified constraints.

        Returns:
        list: The generated genome.
        """

        max_n_input = 6
        max_n_hidden = 1000
        min_lr = 0.001
        max_lr = 1.0
        min_norm = 1
        max_norm = 1000
        # min_bias = -1.0
        # max_bias = 1.0

        n_input = random.randint(1, max_n_input)
        input_indices = random.sample(range(max_n_input), n_input)

        while len(input_indices) < n_input:
            input_indices = random.sample(range(max_n_input), n_input)

        n_hidden = random.randint(1, max_n_hidden)
        learn_rate = random.uniform(min_lr, max_lr)
        norm = random.randint(min_norm, max_norm)
        # bias = random.uniform(min_bias, max_bias)

        input_flags = [0] * max_n_input
        for idx in input_indices:
            input_flags[idx] = 1

        # genome = [n_input] + input_flags + [n_hidden, learn_rate, norm, bias]
        genome = [n_input] + input_flags + [n_hidden, learn_rate, norm]

        return genome

    def fitness_function(self, ga_instance, solution, solution_idx):
        # Decode genome into MLP parameters and architecture
        mlp_params = self.decode_genome(solution)

        # Run Flappy Bird using this MLP and return the score as the fitness
        score = self.run_flappy_bird(
            mlp_params,
            ga_instance.generations_completed,
            self.num_generations,
            solution_idx,
        )
        return score

    def decode_genome(self, genome):
        # Convert 1D genome into MLP parameters and architecture
        n_input = genome[0]
        input_flags = genome[1:7]  # i0 to i5
        n_hidden = genome[7]
        learn_rate = genome[8]
        norm = genome[9]
        # bias = genome[10]

        # Decoding input flags
        input_indices = [idx for idx, flag in enumerate(input_flags) if flag == 1]

        # Example: returning decoded parameters as a dictionary
        decoded_params = {
            "n_input": n_input,
            "input_indices": input_indices,
            "n_hidden": n_hidden,
            "learn_rate": learn_rate,
            "norm": norm,
            # "bias": bias,
        }

        return decoded_params

    def encode_genome(params):
        """
        Encodes MLP parameters into a 1D genome array.

        Args:
        params (dict): A dictionary containing MLP parameters.

        Returns:
        list: The encoded genome.
        """
        # Extracting values from the parameters
        n_input = params["n_input"]
        input_indices = params["input_indices"]
        n_hidden = params["n_hidden"]
        learn_rate = params["learn_rate"]
        norm = params["norm"]
        # bias = params["bias"]

        # Encoding input flags
        input_flags = [0, 0, 0, 0, 0, 0]
        for idx in input_indices:
            input_flags[idx] = 1

        # Constructing the genome
        # genome = [n_input] + input_flags + [n_hidden, learn_rate, norm, bias]
        genome = [n_input] + input_flags + [n_hidden, learn_rate, norm]

        return genome

    def ensure_valid_genomes(self, ga_instance, offspring):
        """
        This function will be called after the mutation operation in each generation
        and will adjust the genomes to ensure they satisfy the problem's constraints.
        """
        new_population = []

        for genome in ga_instance.population:
            # Check and adjust the genome to satisfy the constraints.
            genome[0] = int(round(genome[0]))
            if genome[0] == 0:
                genome[0] = 1
            elif genome[0] > len(self.GENOME):
                genome[0] = 6

            genome[7] = int(round(genome[7]))
            if genome[7] == 0:
                genome[7] = 1

            if genome[8] == 0.0:
                genome[8] = 0.1

            if genome[9] == 0.0:
                genome[9] = 1

            # Ensure input_indices has length n_input
            input_flags = genome[1:7]
            input_flags = [1 if val >= 0.5 else 0 for val in input_flags]

            n_active_flags = int(genome[0])

            while sum(input_flags) != n_active_flags:
                print(sum(input_flags), n_active_flags)
                print(input_flags)
                if sum(input_flags) > n_active_flags:
                    # deactivate a random active flag
                    active_flags = [i for i, x in enumerate(input_flags) if x == 1]
                    try:
                        input_flags[random.choice(active_flags)] = 0
                    except:
                        pass
                else:
                    # activate a random inactive flag
                    inactive_flags = [i for i, x in enumerate(input_flags) if x == 0]
                    try:
                        input_flags[random.choice(inactive_flags)] = 1
                    except:
                        pass

            genome[1:7] = input_flags
            new_population.append(genome)

        # Replace the old population with the new one
        ga_instance.population = np.array(new_population)

    def run_flappy_bird(self, mlp_params, generation, max_generations, solution_idx):
        # Run an instance of Flappy Bird using the provided MLP parameters and architecture
        # and return the score
        score = FlappyMLP.main(
            mlp_params, generation + 1, max_generations + 1, solution_idx + 1
        )

        return score

    def run(self):
        self.ga_instance.run()
        solution, solution_fitness, _ = self.ga_instance.best_solution()
        return solution, solution_fitness


if __name__ == "__main__":
    ga = FlappyBirdGA()
    best_solution, best_fitness = ga.run()
