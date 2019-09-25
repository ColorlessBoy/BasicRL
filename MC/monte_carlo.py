# Policy Evaluation: First Meet Monte Carlo.
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import sys
# path = os.path.split(os.getcwd())[0]
path = os.getcwd()
if path not in sys.path:
    sys.path.append(path) 
from lib.envs.blackjack import BlackjackEnv

def mc_prediction(policy, env, num_episode, gamma=1.0):
    V = defaultdict(float)
    cnt = {}
    for i in range(1, num_episode+1):
        episode = []
        state = env.reset()
        while not env.done:
            action = policy(state)
            next_state, reward, done = env.step(action)
            episode.append([state, action, reward])
            state = next_state
        acc_reward = {}
        r = 0
        for i in range(len(episode)-1, -1, -1):
            r = episode[i][2] + gamma * r
            acc_reward[episode[i][0]] = r # First meet Monte Carlo
        for i in acc_reward:
            if i not in cnt:
                V[i], cnt[i] = acc_reward[i], 1
            else:
                V[i] += acc_reward[i]
                cnt[i] += 1
    for i in V:
        V[i] /= cnt[i]
    return V

def sample_policy(state):
    return 0 if state[0] >= 20 else 1

def plot_value_function(V, title="Value Function"):
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x +1)
    y_range = np.arange(min_y, max_y+1)
    X, Y = np.meshgrid(x_range, y_range)

    # Z requires V is defaultdict.
    Z_noace = np.apply_along_axis(lambda _: V[_[0], _[1], False], 2, np.dstack([X, Y]))
    Z_ace = np.apply_along_axis(lambda _: V[_[0], _[1], True], 2, np.dstack([X, Y]))

    def plot_surface(X, Y, Z, title):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player Sum')
        ax.set_ylabel('Dealer Showing')
        ax.set_zlabel('Value')
        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)
        plt.show()
    
    plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title))
    plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title))

if __name__ == '__main__':
    env = BlackjackEnv()
    steps = 10000
    V = mc_prediction(sample_policy, env, steps)
    plot_value_function(V, title = "Blackjack Game with {} steps".format(steps))
    steps = 500000
    V = mc_prediction(sample_policy, env, steps)
    plot_value_function(V, title = "Blackjack Game with {} steps".format(steps))
    print('Done!')

