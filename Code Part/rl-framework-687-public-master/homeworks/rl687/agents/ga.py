import numpy as np
from .bbo_agent import BBOAgent

from typing import Callable


class GA(BBOAgent):
    """
    A canonical Genetic Algorithm (GA) for policy search is a black box 
    optimization (BBO) algorithm. 
    
    Parameters
    ----------
    populationSize (int): the number of individuals per generation
    numEpisodes (int): the number of episodes to sample per policy         
    evaluationFunction (function): evaluates a parameterized policy
        input: a parameterized policy theta, numEpisodes
        output: the estimated return of the policy            
    initPopulationFunction (function): creates the first generation of
                    individuals to be used for the GA
        input: populationSize (int)
        output: a numpy matrix of size (N x M) where N is the number of 
                individuals in the population and M is the number of 
                parameters (size of the parameter vector)
    numElite (int): the number of top individuals from the current generation
                    to be copied (unmodified) to the next generation
    
    """

    def __init__(self, populationSize:int, evaluationFunction:Callable, 
                 initPopulationFunction:Callable, numElite:int=1, numEpisodes:int=10, Kp = 10, alpha = 2.5):
        #TODO
        self._name = "GeneticAlgorithm"
        self._populationSize = populationSize
        self._numElite = numElite
        self._numEpisodes = numEpisodes
        self._evaluationFunction = evaluationFunction
        self._initPopulationFunction = initPopulationFunction
        self._Kp = Kp
        self._alpha = alpha
        self._population = self._initPopulationFunction(self._populationSize)
        self._placeholder_population = self._population
        self._parameters = None
        self._bestCost = 0

    @property
    def name(self)->str:
        #TODO
        return self._name
    
    @property
    def parameters(self)->np.ndarray:
        #TODO
        return self._parameters

    def _mutate(self, parent:np.ndarray)->np.ndarray:
        """
        Perform a mutation operation to create a child for the next generation.
        The parent must remain unmodified. 
        
        output:
            child -- a mutated copy of the parent
        """
        
        #TODO
        
        epsilon = np.random.randn(parent.size)
        
        return parent + self._alpha * epsilon
    
    def get_parents(self, Kp : int, theta_values : np.ndarray) -> np.ndarray:
        
        return theta_values[0 : Kp, :].copy()
    
    def get_children(self, parents : np.ndarray) -> np.ndarray:
        
        
        
        sampled_parents = parents[np.random.choice(parents.shape[0], self._populationSize - self._numElite), :]
        
        children = np.zeros((self._populationSize - self._numElite, parents.shape[1]))
        
        for i in range(sampled_parents.shape[0]):
            children[i] = self._mutate(sampled_parents[i])
            
        return children

    def train(self)->np.ndarray:
        #TODO
        
        values = []
        
        for k in range(self._population.shape[0]):
#             print(k)
            theta_k = self._population[k, :]
            J_k = self._evaluationFunction(theta_k, self._numEpisodes)
            values.append((theta_k, J_k))
        sorted_values = sorted(values, key = lambda second: second[1], reverse = True)
        theta_values = np.asarray([i[0] for i in sorted_values])
        J_values = np.asarray([i[1] for i in sorted_values])
#         print("New: %f", J_values[0])
        parents = self.get_parents(self._Kp, theta_values)
        elite = theta_values[0 : self._numElite]
#         print("Num Elite: ", end='')
#         print(self._numElite)
#         print("Num Population: ", end='')
#         print(self._populationSize)
#         print("Kp: ", end='')
#         print(self._Kp)
#         print("NumEpisodes: %f", self._numEpisodes)
#         print("Elite shape: ", end = '')
#         print(np.shape(elite))
        self._population = np.vstack((elite, self.get_children(parents)))
#         print("Population shape: ", end = '')
#         print(self._population.shape)
        
        if self._parameters is None or self._bestCost > J_values[0]:
            self._parameters = theta_values[0]
            self._bestCost = J_values[0]
        
        return theta_values[0]

    def reset(self)->None:
        #TODO
        self._population = self._placeholder_population