# Value Iteration: First Meet Monte Carlo
# It's very slow, and I don't know why.
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

def epsilon_greedy_policy(Q, epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype = float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def epsilon_monte_carlo(env, num_episodes, epsilon = 0.1, gamma = 1.0):
    Q = defaultdict(lambda: np.zeros(env.nA, dtype = float))
    values = defaultdict(float)
    cnt = defaultdict(float)
    policy = epsilon_greedy_policy(Q, epsilon, env.nA)
    for i_episode in range(num_episodes):
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        episode = []
        state = env.reset()
        while not env.done:
            probs = policy(state)
            action = np.random.choice(np.arange(env.nA), p = probs)
            next_state, reward, done = env.step(action)
            episode.append((state, action, reward))
            state = next_state
        acc_reward = {}
        r = 0.0
        for i in range(len(episode)-1, -1, -1):
            r = episode[i][2] + gamma * r
            acc_reward[(episode[i][0], episode[i][1])] = r
        for i in acc_reward:
            values[i] += acc_reward[i]
            cnt[i] += 1.0
        for i in cnt:
            Q[i[0]][i[1]] = values[i]/cnt[i]
        # implict: policy = epsilon_greedy_policy(Q, epsilon, env.nA)
    return Q, policy

def get_value_function(Q, policy):
    V = defaultdict(float)
    for (state,actions) in Q.items():
        V[state] = np.max(actions)
    return V

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
    steps = 500000
    Q, policy = epsilon_monte_carlo(env, steps)
    V = get_value_function(Q, policy)
    plot_value_function(V)