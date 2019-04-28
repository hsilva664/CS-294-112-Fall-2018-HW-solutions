import numpy as np
from gym import spaces
from gym import Env
import math


class PointEnv(Env):
    """
    point mass on a 2-D plane
    goals are sampled randomly from a square
    """

    def __init__(self, num_tasks=1, disjoint_sets=False, delta=1):
        self.disjoint_sets = disjoint_sets
        self.delta = delta        
        self.reset_task()
        self.reset()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,))


    def reset_task(self, is_evaluation=False):
        '''
        sample a new task randomly

        Problem 3: make training and evaluation goals disjoint sets
        if `is_evaluation` is true, sample from the evaluation set,
        otherwise sample from the training set
        '''
        #====================================================================================#
        #                           ----------PROBLEM 3----------
        #====================================================================================#
        # YOUR CODE HERE
        if self.disjoint_sets: #Creates checkerboard pattern that goes from -10 to 10 (inclusive), with the length of a tile given by delta. Training only happens on even tiles and testing on odd tiles
            num_groups = math.ceil(21./self.delta)
            even_num_groups = math.ceil(float(num_groups)/2)
            odd_num_groups = math.floor(float(num_groups)/2)

            x_group = np.random.randint(0, num_groups)

            if is_evaluation == False:
                if x_group % 2 == 0: 
                    y_group = 2*np.random.randint(0, even_num_groups) 
                else:
                    y_group = 2*np.random.randint(0, odd_num_groups) + 1
            else:
                if x_group % 2 == 0: 
                    y_group = 2*np.random.randint(0, odd_num_groups) + 1
                else:
                    y_group = 2*np.random.randint(0, even_num_groups)

            delta_x = np.random.uniform(0, self.delta)
            delta_y = np.random.uniform(0, self.delta)

            x = x_group * self.delta + delta_x
            y = y_group * self.delta + delta_y

            x = x if x < 20 else 20
            y = y if y < 20 else 20  

            x = x - 10
            y = y - 10
        else:
            x = np.random.uniform(-10, 10)
            y = np.random.uniform(-10, 10)
        self._goal = np.array([x, y])

    def reset(self):
        self._state = np.array([0, 0], dtype=np.float32)
        return self._get_obs()

    def _get_obs(self):
        return np.copy(self._state)

    def reward_function(self, x, y):
        return - (x ** 2 + y ** 2) ** 0.5

    def step(self, action):
        x, y = self._state
        # compute reward, add penalty for large actions instead of clipping them
        x -= self._goal[0]
        y -= self._goal[1]
        # check if task is complete
        done = abs(x) < .01 and abs(y) < .01
        reward = self.reward_function(x, y)
        # move to next state
        self._state = self._state + action
        ob = self._get_obs()
        return ob, reward, done, dict()

    def viewer_setup(self):
        print('no viewer')
        pass

    def render(self):
        print('current state:', self._state)

    def seed(self, seed):
        np.random.seed = seed
