# The core of GridWorldEnv is the state transition matrix P.

import numpy as np

UP, LEFT, RIGHT, DOWN = 0, 1, 2, 3
class GridWorldEnv():
    def __init__(self, shape=[4,4]):
        self.shape = shape
        self.nS = np.prod(shape)
        self.nA = 4
        maxY = shape[0]
        maxX = shape[1]
        grid = np.arange(self.nS).reshape(shape)
        it = np.nditer(grid, flags=["multi_index"])
        P = {}
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index
            P[s] = {a:[] for a in range(self.nA)}
            is_done = lambda s: s == 0 or s == self.nS-1
            reward = 0.0 if is_done(s) else -1
            if is_done(s):
                P[s][UP] = [1.0, s, reward, True]
                P[s][LEFT] = [1.0, s, reward, True]
                P[s][RIGHT] = [1.0, s, reward, True]
                P[s][DOWN] = [1.0, s, reward, True]
            else:
                s_up = s if y == 0 else s - maxX
                s_left = s if x == 0 else s - 1
                s_right = s if x == maxX - 1 else s + 1
                s_down = s if y == maxY - 1 else s + maxX
                P[s][UP] = [1.0, s_up, reward, is_done(s_up)]
                P[s][LEFT] = [1.0, s_left, reward, is_done(s_left)]
                P[s][RIGHT] = [1.0, s_right, reward, is_done(s_right)]
                P[s][DOWN] = [1.0, s_down, reward, is_done(s_down)]
            it.iternext()
        self.P = P

if __name__ == '__main__':
    env = GridWorldEnv()
    print(P.shape)

