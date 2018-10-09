import numpy as np
from gym import spaces
from gym import Env


class PointEnv(Env):
    """
    point mass on a 2-D plane
    two tasks: move to (-1, -1) or move to (1,1)
    """

    def __init__(self, task={'direction': 1}):
        self.reset_task(task)
        self.reset_model()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,))

    def reset_task(self, task):
        self._task = task
        if task['direction'] == 1:
            self._goal = np.array([1, 1])
        else:
            self._goal = np.array([-1, -1])

    def reset_model(self):
        state = np.random.uniform(-1, 1, size=(2,))
        self._state = np.concatenate([state, np.array([self._task['direction']], dtype=np.float32)])
        return self._get_obs()

    def reset(self):
        return self.reset_model()

    def _get_obs(self):
        return np.copy(self._state)

    def step(self, action):
        self._state = self._state + np.concatenate([action, np.zeros(1, dtype=np.float32)])
        x, y, task = self._state
        x -= self._goal[0]
        y -= self._goal[1]
        reward = - (x ** 2 + y ** 2) ** 0.5
        done = abs(x) < 0.01 and abs(y) < 0.01
        ob = self._get_obs()
        return ob, reward, done, dict()

    def viewer_setup(self):
        print('no viewer')
        pass

    def render(self):
        print('current state:', self._state)

