import numpy as np
import random
import operator
import pandas as pd
import matplotlib.pyplot as plt


def create_new_member(n_city):
    pop = set(np.arange(n_city, dtype=int))
    route = list(random.sample(pop, n_city))

    return route

# primeira geração


def create_starting_population(size, n_city):
    population = []

    for i in range(0, size):
        population.append(create_new_member(n_city))

    return population

# distancia entre cidade i e cidade j


def distance(i, j):
    return np.sqrt((i[0]-j[0])**2 + (i[1]-j[1])**2)

# fitness individual


def fitness(route, cities):
    score = 0
    for i in range(1, len(route)):
        k = int(route[i-1])
        l = int(route[i])

        score = score + distance(cities[k], cities[l])

    return score

# fitness de toda população


def score_population(population, cities):
    scores = []

    for i in population:
        scores.append(fitness(i, cities))

    return scores


def crossover(a, b):
    child = []
    childA = []
    childB = []

    geneA = int(random.random() * len(a))
    geneB = int(random.random() * len(a))

    start_gene = min(geneA, geneB)
    end_gene = max(geneA, geneB)

    # seleciona randomicamente uma parte do primeiro progenitor
    for i in range(start_gene, end_gene):
        childA.append(a[i])

    # insere genes do segundo progenitor
    childB = [item for item in a if item not in childA]
    child = childA+childB

    return child


def breed_population(mating_pool):
    children = []
    for i in range(len(mating_pool)-1):
        children.append(crossover(mating_pool[i], mating_pool[i+1]))
    return children

# duas cidades vão mudar de lugar


def mutate(route, probablity):
    route = np.array(route)
    for swaping_p in range(len(route)):
        if (random.random() < probablity):
            swapedWith = np.random.randint(0, len(route))
            temp1 = route[swaping_p]
            temp2 = route[swapedWith]
            route[swapedWith] = temp1
            route[swaping_p] = temp2

    return route


def selection(popRanked, eliteSize):
    selectionResults = []

    # roulette wheel
    df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()

    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])

    for i in range(0, len(popRanked) - eliteSize):
        pick = 100*random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i, 3]:
                selectionResults.append(popRanked[i][0])
                break

    return selectionResults


def get_all_fitness(population, cities_list):
    fitnessResults = {}
    for i in range(0, len(population)):
        fitnessResults[i] = fitness(population[i], cities_list)
    return sorted(fitnessResults.items(), key=operator.itemgetter(1), reverse=False)

# função mutação para toda a população
def mutate_population(children, mutation_rate):
    new_generation = []
    for i in children:
        mutated_child = mutate(i, mutation_rate)
        new_generation.append(mutated_child)
    return new_generation


def mating(population, selectionResults):
    mating_pool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        mating_pool.append(population[index])
    return mating_pool


def next_generation(cities_list, current_population, mutation_rate, elite_size):
    # rankeia as rotas da próxima geração
    population_rank = get_all_fitness(current_population, cities_list)

    # determina os potenciais progenitores
    selection_result = selection(population_rank, elite_size)

    mating_pool = mating(current_population, selection_result)

    # cria nova geração
    children = breed_population(mating_pool)

    # aplica mutação nela
    next_generation = mutate_population(children, mutation_rate)

    return next_generation


def a_romaria_da_vovo(cities_list, size_population=100, elite_size=20, mutation_rate=0.01, generation=500):
    pop = create_starting_population(size_population, len(cities_list))

    print("Primeira distância de rota encontrada: " +
          str(get_all_fitness(pop, cities_list)[0][1]))

    for i in range(0, generation):
        pop = next_generation(cities_list, pop, mutation_rate, elite_size)
        print("Nova distância de rota: " +
              str(get_all_fitness(pop, cities_list)[0][1]))

    print("Última e melhor distância de rota encontrada: " +
          str(get_all_fitness(pop, cities_list)[0][1]))


cities = []

with open('data.txt') as f:
    f.readline().strip()
    for line in f:
        line = line.strip()
        line = line.split()
        line.pop(2)
        cities.append((float(line[0]),
                       float(line[1])))

a_romaria_da_vovo(cities_list=cities)
