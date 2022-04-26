import operator
import random

import numpy as np
import matplotlib.pyplot as plt

from environment import Environment
from ant import Ant 

# Class representing the ant colony
"""
    ant_population: the number of ants in the ant colony
    iterations: the number of iterations 
    alpha: a parameter controlling the influence of the amount of pheromone during ants' path selection process
    beta: a parameter controlling the influence of the distance to the next node during ants' path selection process
    rho: pheromone evaporation rate
"""
class AntColony:
    def __init__(self, ant_population: int, iterations: int, alpha: float, beta: float, rho: float):
        self.ant_population = ant_population
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho 

        # Initialize the environment of the ant colony
        self.environment = Environment(self.rho)

        # Initilize the list of ants of the ant colony
        self.ants = []

        # Initialize the ants of the ant colony
        for i in range(ant_population):
            
            # Initialize an ant on a random initial location 
            ant = Ant(self.alpha, self.beta, random.choice(self.environment.get_possible_locations()))

            # Position the ant in the environment of the ant colony so that it can move around
            ant.join(self.environment)
        
            # Add the ant to the ant colony
            self.ants.append(ant)

    # Solve the ant colony optimization problem  
    def solve(self):
        shortest_distance = np.inf
        solution = []

        self.environment.initialize_pheromone_map(self.ant_population)

        differences = []
        for i in range(self.iterations):
            distances = []
            best_ant = 0
            best_distance = np.inf
            for j, ant in enumerate(self.ants):
                ant.run()
                distances.append(ant.travelled_distance)

                # Shortest of iteration
                if ant.travelled_distance < best_distance:
                    best_ant = i
                    best_distance = ant.travelled_distance

                # Shortest overall
                if ant.travelled_distance < shortest_distance:
                    shortest_distance = ant.travelled_distance
                    solution = ant.path

            differences.append(max(distances) - min(distances))
            plot(self.environment.coordinates, self.ants[best_ant].path, i)
            print(f'[{i+1:03d}] Shortest distance: {best_distance}')

            self.environment.update_pheromone_map(self.ants)

            if i != self.iterations - 1:
                for ant in self.ants:
                    ant.reset(random.choice(self.environment.get_possible_locations()))

        # plotting
        x = np.arange(1, self.iterations + 1)
        y = np.array(differences)
        plt.plot(x, y)
        plt.show()

        return solution, shortest_distance


def plot(points, path: list, idx=0):
    x = []
    y = []
    for point in points:
        x.append(point[0])
        y.append(point[1])
    y = list(map(operator.sub, [max(y) for i in range(len(points))], y))
    plt.plot(x, y, 'co')

    for _ in range(0, len(path)):
        i = path[_ - 1] - 1
        j = path[_] - 1
        plt.arrow(x[i], y[i], x[j] - x[i], y[j] - y[i], color='r', length_includes_head=True)

    plt.xlim(0, max(x) * 1.1)
    plt.ylim(0, max(y) * 1.1)
    #plt.show()
    plt.savefig(f'plots/path_{idx:04d}.png')
    plt.cla()


def main():
    # Initialize the ant colony
    # ant_colony = AntColony(50, 25, 1, 5, 0.5)   # 12078
    # ant_colony = AntColony(100, 25, 1, 5, 0.5)   # 11175
    # ant_colony = AntColony(100, 25, 1, 2, 0.5)   # 11794 slower convergence towards min
    ant_colony = AntColony(50, 50, 1, 5, 0.3)   # 11141

    # Solve the ant colony optimization problem
    solution, distance = ant_colony.solve()
    print("Solution: ", solution)
    print("Distance: ", distance)

    plot(ant_colony.environment.coordinates, solution)


if __name__ == '__main__':
    main()
