# Policy Iteration

import numpy as np
import sys
import os
path = os.path.split(os.getcwd())[0]
if path not in sys.path:
    sys.path.append(path) 
from lib.envs.gridworld import GridWorldEnv

def policy_eval(policy, env, gamma = 1.0, theta = 0.000001):
    V = np.zeros(env.nS)
    while True:
        delta = 0
        for s in range(env.nS):
            v = 0
            for a, action_prob in enumerate(policy[s]):
                prob, next_state, reward, _ = env.P[s][a]
                v += action_prob * prob * (reward + gamma * V[next_state]) 
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < theta: break
    return V
                
def policy_iteration(env, policy_eval_fn = policy_eval, gamma = 1.0):
    def bellman_operator(s, V):
        A = np.zeros(env.nA)
        for a in range(env.nA):
            prob, next_state, reward, _ = env.P[s][a]
            A[a] += prob * (reward + gamma * V[next_state])
        return A
    policy = np.ones([env.nS, env.nA])/env.nA
    while True:
        V = policy_eval_fn(policy, env, gamma)
        stable = True
        for s in range(env.nS):
            A = bellman_operator(s, V)
            old_action = np.argmax(policy[s])
            new_action = np.argmax(A)
            if old_action != new_action: stable = False
            policy[s] = np.zeros(env.nA)
            policy[s][new_action] = 1
        if stable: break
    return policy, V

env = GridWorldEnv()
policy, V = policy_iteration(env)
print(policy)
print(V)





    