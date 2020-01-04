import numpy as np
from .skeleton import Policy
from typing import Union
from itertools import product

class TabularSoftmaxContinuous(Policy):
    """
    A Tabular Softmax Policy (bs)


    Parameters
    ----------
    numStates (int): the number of states the tabular softmax policy has
    numActions (int): the number of actions the tabular softmax policy has
    """

    def __init__(self, k: int, numActions: int):
        
        #The internal policy parameters must be stored as a matrix of size
        #(numStates x numActions)
        self._k = k
        self._theta = np.zeros((numActions, (self._k + 1) ** 4))
        
        #TODO
        self._numStates = (k + 1) ** 4
        self._numActions = numActions
        self._max = np.array([3, 10, 0.27, 0.45])
        self._min = np.array([-3, -10, -0.27, -0.45])
        # for k=2  
        self._c = np.asarray(list(product(range(self._k + 1), repeat=4)))
        self._phi = None
        self._p = None

    @property
    def parameters(self)->np.ndarray:
        """
        Return the policy parameters as a numpy vector (1D array).
        This should be a vector of length |S|x|A|
        """
        return self._theta.flatten()
    
    @parameters.setter
    def parameters(self, p:np.ndarray):
        """
        Update the policy parameters. Input is a 1D numpy array of size |S|x|A|.
        """
        self._theta = p.reshape(self._theta.shape)

    def __call__(self, state:int, action=None)->Union[float, np.ndarray]:
        
        #TODO
        if action is not None:
            return self.getActionProbabilities(state)[action]
        else:
            return self.getActionProbabilities(state)

    def samplAction(self, state:np.ndarray)->int:
        """
        Samples an action to take given the state provided. 
        
        output:
            action -- the sampled action
        """
        
        #TODO
        
        return np.random.choice(np.arange(self._numActions), p = self.getActionProbabilities(state))
    
    def normalize(self, state:np.ndarray) -> np.ndarray:
        
        return (state - self._min) / (self._max - self._min)
    
    def calculate_phi(self, state : np.ndarray) -> np.ndarray:
        
        phi = np.cos(np.pi * np.dot(self._c, self.normalize(state)))
        
        return phi

    def getActionProbabilities(self, state:np.ndarray)->np.ndarray:
        """
        Compute the softmax action probabilities for the state provided. 
        
        output:
            distribution -- a 1D numpy array representing a probability 
                            distribution over the actions. The first element
                            should be the probability of taking action 0 in 
                            the state provided.
        """

        #TODO
        
        self._phi = self.calculate_phi(state)
        
        self._p = np.dot(self._theta, self._phi)
        
        exp_state_row = np.exp(self._p)
        
        probabilities = exp_state_row / np.sum(exp_state_row)
        
        return probabilities
