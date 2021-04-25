from typing import List

from BitVector import BitVector
import numpy as np
from numpy.random import randint
from copy import deepcopy

from utlis import int_to_bit_vector


class GeneticAlgorithmConfig:

    def __init__(self):
        self.generations_number = None
        self.subject_function = None
        self.fitness_function = None
        self.crossover_probability = None
        self.mutation_probability = None

    def is_configured(self):
        return all([value is not None for value in vars(self).values()])


class GeneticAlgorithmResult:

    class GenerationResult:

        def __init__(self, chromosomes_fits):
            self.chromosomes_fits = chromosomes_fits

        def fits_sum(self):
            return np.sum(self.chromosomes_fits)

        def fits_mean(self):
            return np.mean(self.chromosomes_fits)

    def __init__(self):
        self.generation_results = []

    def add_generation_result(self, generation_result: GenerationResult):
        self.generation_results.append(generation_result)


def generate_chromosomes(lower_bound: int, upper_bound: int, chromosomes_number: int, chromosome_length_in_bits: int):
    return [int_to_bit_vector(randint(lower_bound, upper_bound), chromosome_length_in_bits)
            for _ in range(chromosomes_number)]


def genetic_algorithm(chromosome_list: List[BitVector], configuration: GeneticAlgorithmConfig) -> GeneticAlgorithmResult:

    if configuration.is_configured() == False:
        return GeneticAlgorithmResult()

    chromosomes = [chromosome.deep_copy() for chromosome in chromosome_list]
    config = deepcopy(configuration)

    result = GeneticAlgorithmResult()

    for _ in range(config.generations_number):
        f_values = [config.subject_function(chromosome.intValue()) for chromosome in chromosomes]
        chromosomes_fits = [config.fitness_function(value) for value in f_values]

        result.add_generation_result(GeneticAlgorithmResult.GenerationResult(chromosomes_fits))

        chromosomes = roulette_selection(chromosomes, chromosomes_fits)
        chromosomes = crossover_chromosomes(chromosomes, config.crossover_probability)
        chromosomes = mutate_random_chromosome(chromosomes, config.mutation_probability)

    return result


def roulette_selection(chromosomes: List[BitVector], chromosomes_fits: List[float]):

    fitness_sum = sum(chromosomes_fits)

    roulette_percentages_per_chromosome = [(chromosome_fit / fitness_sum) * 100 for chromosome_fit in chromosomes_fits]
    percentage_sections = []

    percentage_sum = 0.0
    for percentage in roulette_percentages_per_chromosome:
        percentage_sections.append((percentage_sum, percentage_sum + percentage))
        percentage_sum += percentage

    roulette_hits = [np.random.uniform(0, 1) * 100 for _ in range(len(chromosomes))]

    selected_chromosomes = []
    for hit in roulette_hits:
        chromosome_index = 0
        while (percentage_sections[chromosome_index][0] < hit <= (percentage_sections[chromosome_index][1])) is False:
            chromosome_index = chromosome_index + 1

        selected_chromosomes.append(chromosomes[chromosome_index].deep_copy())

    return selected_chromosomes


def crossover_chromosomes(chromosomes: List[BitVector], crossover_probability: float):

    chromosome_len = chromosomes[0].size

    for first_chromosome, second_chromosome in zip(chromosomes[0::2], chromosomes[1::2]):
        if np.random.uniform(0, 1) < crossover_probability:
            crossover_point = np.random.randint(0, chromosome_len)
            temp_swapped_part = first_chromosome[crossover_point:]
            first_chromosome[crossover_point:] = second_chromosome[crossover_point:]
            second_chromosome[crossover_point:] = temp_swapped_part

    return chromosomes


def mutate_random_chromosome(chromosomes: List[BitVector], mutation_probability: float):

    if np.random.uniform(0, 1) < mutation_probability:
        position = np.random.randint(0, len(chromosomes))
        chromosome_len = chromosomes[0].size
        mutation_point = np.random.randint(0, chromosome_len)
        chromosomes[position][mutation_point] = 0 if chromosomes[position][mutation_point] else 1

    return chromosomes

