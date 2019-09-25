import numpy as np

deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


def usable_ace(hand):
    usable = 1 in hand and sum(hand) + 10 <= 21
    return usable

def sumhand(hand):
    if usable_ace(hand):
        return sum(hand) + 10
    return sum(hand)

def is_bust(hand):
    return sumhand(hand) > 21

class BlackjackEnv():
    def __init__(self):
        self.action_space = np.arange(2)
        self.nA = 2
        self.done = False
        self.reset()

    def _get_obs(self):
        return (sumhand(self.player), self.dealer[0], usable_ace(self.player))

    def reset(self):
        self.done = False
        self.dealer = [np.random.choice(deck), np.random.choice(deck)]
        self.player = [np.random.choice(deck), np.random.choice(deck)]
        while sumhand(self.player) < 12:
            self.player.append(np.random.choice(deck))
        return self._get_obs()

    def step(self, action):
        assert action in self.action_space
        if action: # hit
            self.player.append(np.random.choice(deck))
            if is_bust(self.player):
                self.done = True
                reward = -1.0
            else:
                self.done = False
                reward = 0.0
        else: # stick
            self.done = True
            while sumhand(self.dealer) < 17:
                self.dealer.append(np.random.choice(deck))
            reward = 1.0 if is_bust(self.dealer) \
                or sumhand(self.dealer) < sumhand(self.player) else -1.0
        return self._get_obs(), reward, self.done

if __name__ == '__main__':
    env = BlackjackEnv()
    for _ in range(20):
        env.reset()
        while not env.done:
            action = np.random.choice(env.action_space)
            state, reward, done = env.step(action)
            score, dealer_score, usable = state
            print("Action: {}. Player Score: {} (Usable Ace: {}), Dealer Score: {}".format(
                ['stick', 'hit'][action], score, usable, dealer_score))
            if done:
                print('Game end. Reward: {}\n'.format(reward))