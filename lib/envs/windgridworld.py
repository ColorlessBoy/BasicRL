import numpy as np
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
class WindyGridworldEnv():
    def _cut(self, coord):
        coord[0] = max(0, coord[0])
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[1] = max(0, coord[1])
        coord[1] = min(coord[1], self.shape[1] - 1)
        return coord
    def _next(self, pos, action):
        next_pos = np.array(pos) + np.array(action) + np.array([-1, 0]) * self.winds[tuple(pos)]
        next_pos = _cut(next_pos)
        is_done = tuple(next_pos) == self.terminal
        return (1.0, next_pos, -1.0, is_done)

    def __init__(self):
        self.shape = [7, 10]
        self.nS = np.prod(self.shape)
        self.nA = 4
        self.action_space = np.arange(self.nA)
        self.winds = np.zeros(self.shape)
        self.winds[:, [3,4,5,8]] = 1
        self.winds[:, [6, 7]] = 2
        self.terminal = [3, 7]
        self.P = {}
        for s in range(self.nS):
            pos = (s // self.shape[0], s % self.shape[1])
            P[s] = np.arange(self.nA)
            P[s][UP] = self._next(pos, [-1, 0])
            P[s][RIGHT] = self._next(pos, [0, 1])
            P[s][DOWN] = self._next(pos, [1, 0])
            P[s][LEFT] = self._next(pos, [0, -1])
        self.reset()

    def reset(self):
        self.cur = (3, 0)
        self.done = False
        return (self.cur, 0, self.cur, self.done)
    
    def step(self, action):
        assert action in self.action_space
        pos = self.cur
        _, self.cur, _, self.done = P[self.cur][action]
        return (pos, -1, self.cur, self.done)

if __name__ == '__main__':
    env = WindyGridworldEnv()
    while not env.done:
        pass
