from math import pi, sin
from genetic_algorithm import genetic_algorithm, generate_chromosomes, GeneticAlgorithmResult, GeneticAlgorithmConfig
import matplotlib.pyplot as plt
import pandas as pd
import os


def f(x):
    return 0.2*pow(x, 0.5) + 2*sin(2 * pi * 0.02 * x) + 5


def fitness_function(x):
    return pow(x, 2)


chromosome_length_in_bits = 8

config = GeneticAlgorithmConfig()
config.generations_number = 200
config.subject_function = f
config.fitness_function = fitness_function

crossover_probabilities = [0.5, 0.6, 0.7, 0.8, 1]
mutation_probabilities = [0, 0.01, 0.06, 0.1, 0.2, 0.3, 0.5]

results_path = "results"
if not os.path.exists(results_path):
    os.makedirs(results_path)

for chromosomes_number in [50, 200]:
    mean_values = []
    chromosomes = generate_chromosomes(0, 255, chromosomes_number, chromosome_length_in_bits)

    for mutation_probability in mutation_probabilities:
        mean_values_for_mutation_probability = []
        config.mutation_probability = mutation_probability

        for crossover_probability in crossover_probabilities:
            config.crossover_probability = crossover_probability
            results = genetic_algorithm(chromosomes, config)
            sum_per_gen = [gen_result.fits_sum() for gen_result in results.generation_results]
            last_gen_mean_fit = results.generation_results[chromosomes_number-1].fits_mean()
            mean_values_for_mutation_probability.append(last_gen_mean_fit)
            plt.plot(range(config.generations_number), sum_per_gen, linewidth=0.3, label=f"cp = {crossover_probability}")

        mean_values.append(mean_values_for_mutation_probability)
        plt.ylabel("Fitness function sum")
        plt.xlabel("Generation number")
        plt.legend(title="legend", loc="upper left", bbox_to_anchor=(1.05, 1))
        plt.title(f"mutation probability = {mutation_probability}")
        plt.savefig(f"{results_path}/plot {chromosomes_number} {mutation_probability}.jpg", bbox_inches="tight")
        plt.clf()

    df = pd.DataFrame(mean_values, index=mutation_probabilities, columns=crossover_probabilities)
    df.to_excel(f"{results_path}/output {chromosomes_number}.xlsx")
