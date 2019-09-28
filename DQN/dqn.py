import itertools
import gym
import numpy as np
import tensorflow as tf

from models import StateProcessor, Estimator, ModelParametersCopier
from replay_buffer import ReplayBuffer

def make_epsilon_greedy_policy(estimator, nA):
    def policy_fn(sess, observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def deep_q_learning(env,
                    sess,
                    q_estimator,
                    target_estimator,
                    state_processor,
                    num_episodes,
                    replay_memory_size = 500000,
                    update_target_estimator_every = 10000,
                    discount_factor = 0.99,
                    epsilon_start = 1.0,
                    epsilon_end = 0.1,
                    epsilon_decay_steps=500000,
                    batch_size = 32):

    memory = ReplayBuffer(replay_memory_size)
    estimator_copy = ModelParametersCopier(q_estimator, target_estimator)

    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
    policy = make_epsilon_greedy_policy(
        q_estimator,
        env.action_space.n
    )

    for i_episode in range(num_episodes):
        state = env.reset()
        state = state_processor.process(sess, state)
        state = np.stack([state] * batch_size, axis = 2)
        loss = None
        stats = []
        for t in itertools.count():
            epsilon = epsilons[min(t, epsilon_decay_steps-1)]

            if t % update_target_estimator_every == 0:
                estimator_copy.make(sess)
                print("\nUpdate target network.")

            action_probs = policy(sess, state, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)
            next_state = state_processor.process(sess, next_state)
            next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
            memory.add(next_state)

            states_batch, action_batch, reward_batch, next_state_batch, done_batch = \
                memory.sample(batch_size)
            q_values_next = target_estimator.predict(sess,next_state_batch)
            target_batch = reward_batch + np.invert(done_batch).astype(np.float32) \
                * discount_factor * np.amax(q_values_next, axis=1)
            print(target_batch)
            loss = q_estimator.update(sess, states_batch, action_batch, target_batch)

            if done: 
                stats.append(t)
                break

            state = next_state
            yield t, stats
    return stats

if __name__ == "__main__":
    env = gym.envs.make('Breakout-v0')
    q_estimator = Estimator(scope = "q_estimator")
    target_estimator = Estimator(scope = "target_q")
    state_processor = StateProcessor()
    with tf.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        for t, stats in deep_q_learning(sess,
                                        env,
                                        q_estimator=q_estimator,
                                        target_estimator=target_estimator,
                                        state_processor=state_processor,
                                        num_episodes=10000):
            print("\nEpisode Length: {}".format(stats[-1]))
