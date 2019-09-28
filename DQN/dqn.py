import itertools
import gym
import numpy as np
import skimage
import tensorflow as tf

from models import Estimator
from replay_buffer import ReplayBuffer

def state_process(state):
    state = skimage.color.rgb2gray(state)
    state = state[34:, :]
    state = skimage.transform.resize(state, (84, 84))
    state = state / 255.0
    return np.stack([state] * 4, axis=2)

def make_epsilon_greedy_policy(estimator, nA):
    def policy_fn(observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(np.expand_dims(observation, 0))
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def deep_q_learning(env,
                    q_estimator,
                    target_estimator,
                    num_episodes,
                    replay_memory_size = 5000,
                    update_target_estimator_every = 100,
                    discount_factor = 0.99,
                    epsilon_start = 1.0,
                    epsilon_end = 0.1,
                    epsilon_decay_steps=5000,
                    batch_size = 32):

    memory = ReplayBuffer(replay_memory_size)

    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
    policy = make_epsilon_greedy_policy(
        q_estimator,
        env.action_space.n
    )

    for i_episode in range(num_episodes):
        state = env.reset()
        state = state_process(state)
        stats = []
        for t in itertools.count():
            epsilon = epsilons[min(t, epsilon_decay_steps-1)]

            if t % update_target_estimator_every == 0:
                # target_estimator.save()
                # q_estimator.load()
                target_estimator.copy(q_estimator)
                print("\nUpdate target network.")

            action_probs = policy(state, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)
            next_state = state_process(next_state)
            memory.add(state, action, reward, next_state, done)

            states_batch, action_batch, reward_batch, next_state_batch, done_batch = \
                memory.sample(batch_size)
            q_values_next = target_estimator.predict(next_state_batch)
            target_batch = reward_batch + np.invert(done_batch).astype(np.float32) \
                * discount_factor * np.amax(q_values_next, axis=1)
            q_estimator.update(states_batch, action_batch, target_batch)

            if done: 
                print('\nEpisode {} ends in {} steps'.format(len(stats), t))
                stats.append(t)
                break

            state = next_state
    return stats

if __name__ == "__main__":
    env = gym.envs.make('Breakout-v0')
    q_estimator = Estimator()
    target_estimator = Estimator()
    deep_q_learning(env,
                    q_estimator=q_estimator,
                    target_estimator=target_estimator,
                    num_episodes=10000)
