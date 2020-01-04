import numpy as np
from typing import Tuple
from .skeleton import Environment


class Cartpole(Environment):
    """
    The cart-pole environment as described in the 687 course material. This
    domain is modeled as a pole balancing on a cart. The agent must learn to
    move the cart forwards and backwards to keep the pole from falling.

    Actions: left (0) and right (1)
    Reward: 1 always

    Environment Dynamics: See the work of Florian 2007
    (Correct equations for the dynamics of the cart-pole system) for the
    observation of the correct dynamics.
    """

    def __init__(self):
        # TODO: properly define the variables below
        self._name = "Cartpole"
        self._action = None
        self._reward = 1.0
        self._isEnd = False
        self._gamma = 1.0
        self._actions = [0, 1]

        # define the state # NOTE: you must use these variable names
        self._x = 0.  # horizontal position of cart
        self._v = 0.  # horizontal velocity of the cart
        self._theta = 0.  # angle of the pole
        self._dtheta = 0.  # angular velocity of the pole

        # dynamics
        self._g = 9.8  # gravitational acceleration (m/s^2)
        self._mp = 0.1  # pole mass
        self._mc = 1.0  # cart mass
        self._l = 0.5  # (1/2) * pole length
        self._dt = 0.02  # timestep
        self._t = 0.0  # total time elapsed  NOTE: USE must use this variable
        
        self._state = np.array([self._x, self._v, self._theta, self._dtheta])
        
        self._action_count = len(self._actions)

    @property
    def name(self) -> str:
        # TODO
        return self._name

    @property
    def reward(self) -> float:
        # TODO
        return self._reward

    @property
    def gamma(self) -> float:
        # TODO
        return self._gamma

    @property
    def action(self) -> int:
        # TODO
        return self._action

    @property
    def isEnd(self) -> bool:
        # TODO
        return self.terminal() == True

    @property
    def state(self) -> np.ndarray:
        # TODO
        return self._state

    def nextState(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        Compute the next state of the pendulum using the euler approximation to the dynamics
        """
        # TODO
        
        _x, _v, _theta, _dtheta = state
        
        F = 10
        
        if(action == 0):
            F = -10
            
        _ddtheta = (self._g * np.sin(_theta) + np.cos(_theta) * ((-F - self._mp * self._l * (_dtheta ** 2) * np.sin(_theta)) / (self._mc + self._mp))) / (self._l * ((4.0/3.0)  - (self._mp * (np.cos(_theta) ** 2) / (self._mp + self._mc))))
        
        _ddx = (F + self._mp * self._l * ((_dtheta ** 2) * np.sin(_theta) - _ddtheta * np.cos(_theta))) / (self._mc + self._mp)
        
        _dx_t = _v
        _dv_t = _ddx
        _dtheta_t = _dtheta
        _domega_t = _ddtheta
        
        state += self._dt * np.array([_dx_t, _dv_t, _dtheta_t, _domega_t])
        
        return state
    
            

    def R(self, state: np.ndarray, action: int, nextState: np.ndarray) -> float:
        # TODO
        return self._reward

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        takes one step in the environment and returns the next state, reward, and if it is in the terminal state
        """
        # TODO
        self._action = action
        nextState = self.nextState(self._state, action)
        self._t += self._dt
        
        reward = self.R(self._state, action, nextState)
        
        self._state = nextState
        

        self._isEnd = self.terminal() == True
        
        if(self._isEnd == True):
            return self._state, reward, self._isEnd
        
        self._x, self._v, self._theta, self._dtheta = self._state
        
        self._isEnd = self.terminal() == True
        
        return self._state, reward, self._isEnd

    def reset(self) -> None:
        """
        resets the state of the environment to the initial configuration
        """
        # TODO
        self._x = 0.  # horizontal position of cart
        self._v = 0.  # horizontal velocity of the cart
        self._theta = 0.  # angle of the pole
        self._dtheta = 0.  # angular velocity of the pole
        
        
        self._state = np.array([self._x, self._v, self._theta, self._dtheta])
        
        self._isEnd = False
        self._action = None
        self._reward = 1.0
        self._t = 0.0

    def terminal(self) -> bool:
        """
        The episode is at an end if:
            time is greater that 20 seconds
            pole falls |theta| > (pi/12.0)
            cart hits the sides |x| >= 3
        """
        # TODO
        if((self._t > 20.0) or (self._theta > np.pi / 12.0) or (self._theta < - np.pi / 12.0) or (self._x >= 3.0 or self._x <= -3.0)):
            return True
        else:
            return False
