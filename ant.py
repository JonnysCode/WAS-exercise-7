import random


# Class representing an artificial ant of the ant colony
"""
    alpha: a parameter controlling the influence of the amount of pheromone during ants' path selection process
    beta: a parameter controlling the influence of the distance to the next node during ants' path selection process
"""
class Ant:
    def __init__(self, alpha: float, beta: float, initial_location: int):
        self.alpha = alpha
        self.beta = beta
        self.current_location = initial_location
        self.travelled_distance = 0
        self.path = [initial_location]
        self.pheromone_delta = None

    # The ant runs to visit all the possible locations of the environment 
    def run(self):
        while len(self.path) < len(self.environment.get_possible_locations()):
            next_location = self.select_path()
            self.travelled_distance += self.environment.get_distance(self.current_location, next_location)
            self.current_location = next_location
            self.path.append(next_location)

        self.travelled_distance += self.environment.get_distance(self.current_location, self.path[0])

    # Select the next path based on the random proportional rule of the ACO algorithm
    def select_path(self):
        cities = []
        probabilities = []
        divisor = 0
        for city in list(set(self.environment.get_possible_locations()).difference(set(self.path))):
            value = self._get_probability(city)
            cities.append(city)
            probabilities.append(value)
            divisor += value

        for i, prob in enumerate(probabilities):
            probabilities[i] = prob / divisor if divisor > 0.0 else 1 / len(probabilities)

        return random.choices(cities, probabilities)[0]

    # Position an ant in an environment
    def join(self, environment):
        self.environment = environment

    def _get_probability(self, city):
        tau = self.environment.pheromone[self.current_location - 1][city - 1]
        eta = 1 / self.environment.get_distance(self.current_location, city)

        return (tau ** self.alpha) * (eta ** self.beta)

    def reset(self, location):
        self.current_location = location
        self.travelled_distance = 0
        self.path = [location]
