# a romaria da vovó

This code is an implementation of a genetic algorithm to solve the Traveling Salesman Problem (TSP). The TSP is a classic optimization problem that consists of finding the shortest possible route that visits all given cities and returns to the starting city. The algorithm creates an initial population of random routes, and then evolves this population through successive generations of selection, crossover, and mutation until an optimal solution is found.

The code imports numpy, random, operator, pandas, and matplotlib.pyplot modules. Then, it defines several functions:

- create_new_member: creates a new route (list of cities to visit) for an individual in the population.
- create_starting_population: creates the starting population by calling the create_new_member function a specified number of times.
- distance: calculates the distance between two cities.
- fitness: calculates the fitness (distance) of an individual route.
- score_population: calculates the fitness of each individual in the population.
- crossover: performs a crossover operation on two routes to create a new route.
- breed_population: creates a new population by performing crossover on pairs of routes.
- mutate: performs a mutation operation on a route, swapping the positions of two cities.
- selection: selects individuals from the population for the next generation based on their fitness.
- get_all_fitness: calculates the fitness of each individual in the population and ranks them by fitness.
- mutate_population: performs a mutation operation on each individual in the population.
- mating: selects individuals from the population for the mating pool based on the results of the selection function.
- next_generation: creates the next generation of the population by performing selection, crossover, and mutation.
- a_romaria_da_vovo: a function that initializes the population, evolves the population for a specified number of generations, and prints the distance of the best route found.

Finally, the code reads a list of cities from a file and calls the a_romaria_da_vovo function to solve the TSP for the given cities.
