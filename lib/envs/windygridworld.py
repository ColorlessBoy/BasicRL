import numpy as np
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3

class WindyGridWorldEnv():
    def _cut(self, coord):
        if coord[0] < 0: coord[0] = 0
        if coord[0] >= self.shape[0]: coord[0] = self.shape[0] - 1
        if coord[1] < 0: coord[1] = 0
        if coord[1] >= self.shape[1]: coord[1] = self.shape[1] - 1
        return coord

    def _next(self, pos, action):
        next_pos = np.array(pos) + np.array(action) + np.array([-1, 0]) * self.winds[tuple(pos)]
        next_pos = self._cut(next_pos).astype(int)
        next_pos = np.ravel_multi_index(tuple(next_pos), self.shape)
        is_done = next_pos == self.terminal
        return (1.0, next_pos, -1.0, is_done)

    def __init__(self):
        self.shape = [7, 10]
        self.nS = np.prod(self.shape)
        self.nA = 4
        self.action_space = np.arange(self.nA)
        self.winds = np.zeros(self.shape)
        self.winds[:, [3,4,5,8]] = 1
        self.winds[:, [6, 7]] = 2
        self.terminal = 26
        self.P = {}
        for s in range(self.nS):
            pos = (s // self.shape[1], s % self.shape[1])
            self.P[s] = { a: [] for a in range(self.nA)}
            self.P[s][UP] = self._next(pos, [-1, 0])
            self.P[s][RIGHT] = self._next(pos, [0, 1])
            self.P[s][DOWN] = self._next(pos, [1, 0])
            self.P[s][LEFT] = self._next(pos, [0, -1])
        self.reset()

    def reset(self):
        self.cur = 20
        self.done = False
        return (self.cur, 0, self.cur, self.done)
    
    def step(self, action):
        assert action in self.action_space
        pos = self.cur
        _, self.cur, _, self.done = self.P[self.cur][action]
        return (pos, -1, self.cur, self.done)

if __name__ == '__main__':
    env = WindyGridWorldEnv()
    state = env.reset()
    print(state)
    while not env.done:
        state = env.step(np.random.choice(env.action_space))
        print(state)
        