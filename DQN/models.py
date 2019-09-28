import gym
import numpy as np
import tensorflow as tf

VALID_ACTIONS = [0, 1, 2, 3]

class Estimator():
    def __init__(self, estimator = None):
        if estimator:
            self.model = tf.keras.models.clone_model(estimator.model)
        else:
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
        target_q = self.model.predict(states)
        for idx, action in enumerate(actions):
            target_q[idx][action] = td_targets[idx]
        self.model.fit(states, target_q, epochs = 1, verbose = 0)
    
    def copy(self, estimator):
        self.model = tf.keras.models.clone_model(estimator.model)

if __name__ == "__main__":
    e = Estimator()
    observations = np.random.rand(2, 84, 84, 4)
    print(e.predict(observations))
    # Test training step
    td_target = np.array([10.0, 10.0])
    action = np.array([1, 3])
    e.update(observations, action, td_target)