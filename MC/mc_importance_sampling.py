# First Meet Monte Carlo with weighted important sampling
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys
import os
path = os.getcwd()
if path not in sys.path:
    sys.path.append(path)
from lib.envs.blackjack import BlackjackEnv

def get_random_policy(env):
    A = np.ones(env.nA) / env.nA
    def policy_fn(state):
        return A
    return policy_fn

def get_greedy_policy(Q):
    def policy_fn(state):
        return np.argmax(Q[state])
    return policy_fn

def monte_carlo_weighted_importance_sampling(num_episodes, env, behavior_policy, gamma = 1.0):
    Q = defaultdict(lambda: np.zeros(env.nA))
    C = defaultdict(lambda: np.zeros(env.nA))
    target_policy = get_greedy_policy(Q)
    for i_episode in range(num_episodes):
        if i_episode % 1000 == 0:
            print("\r episode: {}/{}".format(i_episode, num_episodes), end="")
        state = env.reset()
        episodes = []
        while True:
            prob = behavior_policy(state)
            action = np.random.choice(env.action_space, p=prob)
            next_state, reward, done = env.step(action)
            episodes.append((state, action, reward))
            if done: break
            state = next_state      
        acc_rewards = {}
        r = 0.0
        w = 1.0
        for (state, action, reward) in episodes[::-1]:
            r = reward + gamma * r
            acc_rewards[(state, action)] = (w, r)
            if target_policy(state) != action: break # Greedy policy makes the rest w = 0
            w *= (1.0 / behavior_policy(state)[action])
        for (state, action) in acc_rewards:
            w, r = acc_rewards[(state, action)]
            C[state][action] += w
            Q[state][action] += w / C[state][action] * (r - Q[state][action])
    return Q, target_policy

def get_value_function(Q, policy): # greedy policy
    V = defaultdict(float)
    for (state, actions) in Q.items():
        V[state] = actions[policy(state)]
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
    behavior_policy = get_random_policy(env)
    steps = 500000
    Q, target_policy = monte_carlo_weighted_importance_sampling(steps, env, behavior_policy)
    V = get_value_function(Q, target_policy)
    plot_value_function(V)
    print('done')
