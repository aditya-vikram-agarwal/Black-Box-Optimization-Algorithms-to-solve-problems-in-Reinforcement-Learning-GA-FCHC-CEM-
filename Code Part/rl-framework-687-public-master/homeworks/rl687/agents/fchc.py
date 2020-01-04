import numpy as np
from .bbo_agent import BBOAgent

from typing import Callable


class FCHC(BBOAgent):
    """
    First-choice hill-climbing (FCHC) for policy search is a black box optimization (BBO)
    algorithm. This implementation is a variant of Russell et al., 2003. It has 
    remarkably fewer hyperparameters than CEM, which makes it easier to apply. 
    
    Parameters
    ----------
    sigma (float): exploration parameter 
    theta (np.ndarray): initial mean policy parameter vector
    numEpisodes (int): the number of episodes to sample per policy
    evaluationFunction (function): evaluates a provided policy.
        input: policy (np.ndarray: a parameterized policy), numEpisodes
        output: the estimated return of the policy 
    """

    def __init__(self, theta:np.ndarray, sigma:float, evaluationFunction:Callable, numEpisodes:int=10):
        #TODO
        self._name = "Firstchoicehillclimbing"
        self._theta = theta
        self._sigma = sigma
        self._numEpisodes = numEpisodes
        self._placeholder = theta
        self._evaluationFunction = evaluationFunction
        self._J = self._evaluationFunction(self._theta, self._numEpisodes)

    @property
    def name(self)->str:
        #TODO
        return self._name
    
    @property
    def parameters(self)->np.ndarray:
        #TODO
        return self._theta.flatten()

    def train(self)->np.ndarray:
        #TODO

        theta_dash = np.random.multivariate_normal(self.parameters, self._sigma * np.eye(self.parameters.size))
        J_dash = self._evaluationFunction(theta_dash, self._numEpisodes)
        if J_dash > self._J:
            self._theta = theta_dash
            self._J = J_dash
            
        #print("New: %f, J: %f " % (J_dash, self._J))
        
        return self._theta

    def reset(self)->None:
        #TODO
        self._theta = self._placeholder
        self._J = -float("inf")
