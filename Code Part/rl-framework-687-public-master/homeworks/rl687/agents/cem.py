import numpy as np
from .bbo_agent import BBOAgent

from typing import Callable


class CEM(BBOAgent):
    """
    The cross-entropy method (CEM) for policy search is a black box optimization (BBO)
    algorithm. This implementation is based on Stulp and Sigaud (2012). Intuitively,
    CEM starts with a multivariate Gaussian dsitribution over policy parameter vectors.
    This distribution has mean thet and covariance matrix Sigma. It then samples some
    fixed number, K, of policy parameter vectors from this distribution. It evaluates
    these K sampled policies by running each one for N episodes and averaging the
    resulting returns. It then picks the K_e best performing policy parameter
    vectors and fits a multivariate Gaussian to these parameter vectors. The mean and
    covariance matrix for this fit are stored in theta and Sigma and this process
    is repeated.

    Parameters
    ----------
    sigma (float): exploration parameter
    theta (numpy.ndarray): initial mean policy parameter vector
    popSize (int): the population size
    numElite (int): the number of elite policies
    numEpisodes (int): the number of episodes to sample per policy
    evaluationFunction (function): evaluates the provided parameterized policy.
        input: theta_p (numpy.ndarray, a parameterized policy), numEpisodes
        output: the estimated return of the policy
    epsilon (float): small numerical stability parameter
    """

    def __init__(self, theta:np.ndarray, sigma:float, popSize:int, numElite:int, numEpisodes:int, evaluationFunction:Callable, epsilon:float=0.0001):
        #TODO
        self._name = "Cem"
        self._theta = theta
        self._Sigma = sigma * np.eye(self._theta.size)
        self._popSize = popSize
        self._numElite = numElite
        self._numEpisodes = numEpisodes
        self._epsilon = epsilon
        self._placeholder_theta = theta
        self._placeholder_Sigma = self._Sigma
        self._evaluationFunction = evaluationFunction
        

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
        
        values = []
        for k in range(1, self._popSize + 1, 1):
            theta_k = np.random.multivariate_normal(self.parameters, self._Sigma)
            J_dash = self._evaluationFunction(theta_k, self._numEpisodes)
            values.append((theta_k, J_dash))
        sorted_values = sorted(values, key = lambda second: second[1], reverse = True)
        
        theta_values = np.asarray([i[0] for i in sorted_values])[0 : self._numElite]
        J_values = np.asarray([i[1] for i in sorted_values])[0 : self._numElite]

        self._theta = np.sum(theta_values, axis = 0) / self._numElite
        
        new_J = np.sum(J_values, axis = 0) / self._numElite
            
        #print("New: %f" % (new_J, ))

        dot_theta = 0
        for i in range(self._numElite):
            dot_theta += np.dot(np.reshape(theta_values[i] - self._theta, (-1, 1)), np.reshape(np.transpose(theta_values[i] - self._theta), (1, -1)))

        self._Sigma = (self._epsilon * np.eye(self._Sigma.shape[0]) + dot_theta) / (self._epsilon + self._numElite)
            
        return theta_values[0]

    def reset(self)->None:
        #TODO
        self._theta = self._placeholder_theta
        self._Sigma = self._placeholder_Sigma
