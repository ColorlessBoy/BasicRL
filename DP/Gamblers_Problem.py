# Value Iteration

import numpy as np
import sys
import matplotlib.pyplot as plt

def value_iteration_for_gamblers(p_h, target_stakes=50, theta=0.0001, discount_factor=1.0):
    rewards = np.zeros(target_stakes+1)
    rewards[-1] = 1
    V = np.zeros(target_stakes+1)
    def bellman_operator(s, V, rewards):
        A = np.zeros(target_stakes+1)
        max_stakes = min(s, target_stakes-s)
        for a in range(1, max_stakes+1):
            A[a] = p_h * (rewards[s+a] + discount_factor * V[s+a]) \
                + (1 - p_h) * (rewards[s-a] + discount_factor * V[s-a])
        return A
    # Value Iteration
    while True:
        delta = 0
        for s in range(1, target_stakes+1):
            A = bellman_operator(s, V, rewards)
            opt_Vs = np.max(A)
            delta = max(delta, abs(opt_Vs - V[s]))
            V[s] = opt_Vs
        if delta < theta: break
    # Get Policy; Terminal state needs no action.
    policy = np.zeros(target_stakes)
    for s in range(1, target_stakes):
        A = bellman_operator(s,V,rewards)
        opt_As = np.argmax(A)
        policy[s] = opt_As
    return policy, V

target_stakes = 50
policy, V = value_iteration_for_gamblers(0.25)

plt.figure(1)
plt.plot(range(target_stakes+1), V)
plt.xlabel('State')
plt.ylabel('Value')
plt.show()

plt.figure(2)
plt.bar(range(target_stakes), policy, align='center', alpha=0.5)
plt.xlabel('State')
plt.ylabel('Action')
plt.show()