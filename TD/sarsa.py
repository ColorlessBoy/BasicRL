from collections import defaultdict
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
path = os.getcwd()
if path not in sys.path:
    sys.path.append(path)
from lib.envs.windygridworld import WindyGridWorldEnv

def get_epsilon_greedy_policy(Q, nA, epsilon = 0.1):
    def policy_fn(state):
        A = np.ones(nA, dtype = int) * epsilon / nA
        best_action = np.argmax(Q[state])
        A[best_action] += 1 - epsilon
        return A
    return policy_fn

def get_greedy_policy(Q):
    def policy_fn(state):
        return np.argmax(Q[state])
    return policy_fn

def get_Q_score(env, Q):
    _, _, state, _ = env.reset()
    score = 0
    greedy_policy = get_greedy_policy(Q)
    while score < 1000 and not env.done:
        _, _, state, _ = env.step(greedy_policy(state))
        score += 1
    return score

def sarsa(env, num_episodes, epsilon = 0.1, gamma = 1.0, alpha = 0.5):
    Q = defaultdict(lambda: np.zeros(env.nA))
    behavior_policy = get_epsilon_greedy_policy(Q, env.nA, epsilon)
    stats = np.zeros(num_episodes)
    for i_episode in range(num_episodes):
        print("\repoch({}/{})".format(i_episode, num_episodes), end=" ")
        state = env.reset()[0]
        prob = behavior_policy(state)
        action = np.random.choice(env.action_space, p = prob)
        while not env.done:
            _, reward, next_state, done = env.step(action)
            stats[i_episode] += 1

            prob = behavior_policy(next_state)
            next_action = np.random.choice(env.action_space, p = prob)

            td_target = reward + gamma * Q[next_state][next_action]
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error

            # behavior_policy updates implicitly
            state, action = next_state, next_action

    return Q, stats

def plot_episode_stats(stats):
    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(stats)
    plt.show(fig1)

if __name__ == '__main__':
    env = WindyGridWorldEnv()
    steps = 1000
    Q, stats = sarsa(env, steps)
    plot_episode_stats(stats)
    print(get_Q_score(env, Q))
    pass