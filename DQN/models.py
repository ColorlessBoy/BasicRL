import gym
import numpy as np
import skimage
import tensorflow as tf

VALID_ACTIONS = [0, 1, 2, 3]

class Estimator():
    def __init__(self):
        self._build_model()
    
    def _build_model(self):
        self.model = tf.keras.Sequential([
            tf.keras.Input(shape=(84, 84, 4)),
            tf.keras.layers.Conv2D(32, 8, 4, activation='relu'),
            tf.keras.layers.Conv2D(64, 4, 2, activation='relu'),
            tf.keras.layers.Conv2D(64, 3, 1, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512),
            tf.keras.layers.Dense(len(VALID_ACTIONS))
        ])

        optimizer = tf.keras.optimizers.RMSprop(0.00025)
        self.model.compile(loss='mse',
                        optimizer=optimizer,
                        metrics=['mae', 'mse'])
    
    def predict(self, state):
        return self.model.predict(state)

    def update(self, states, actions, td_targets):
        target_q = self.model(states)
        print(target_q)
        # loss = self.model.fit(s, target_q)
        # return loss

def state_process(state):
    state = skimage.color.rgb2gray(state)
    state = state[34:, :]
    state = skimage.transform.resize(state, (84, 84))
    state = state / 255.0
    return state

if __name__ == "__main__":
    env = gym.envs.make('Breakout-v0')
    e = Estimator()
    observation = env.reset()
    print(observation.shape)
    observation = state_process(observation)
    print(observation.shape)
    observation = np.stack([observation]*4, axis=2)
    observations = np.array([observation]*2)

    print(e.predict(observations))
    # Test training step
    y = np.array([10.0, 10.0])
    a = np.array([1, 3])
    print("Successful!")