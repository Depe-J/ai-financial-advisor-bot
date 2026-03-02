import numpy as np
import pandas as pd

# Q-learning trading agent
# state space is based on the relative position of the short and long moving averages
# kept the state space simple on purpose - easier to debug and interpret than a continuous version

class QLearningTrader:
    # initialise the agent with hyperparameters
    # tuned these values through trial and error, alpha=0.3 seemed to converge faster than 0.1
    def __init__(self, bins=10, alpha=0.3, gamma=0.9, epsilon=0.2, seed=42):
        self.bins = bins
        self.alpha = alpha      # learning rate - how fast we update Q values
        self.gamma = gamma      # discount factor - how much we care about future rewards
        self.epsilon = epsilon  # exploration rate - chance of picking a random action
        self.q_table = {}       # stores state -> action values, starts empty
        np.random.seed(seed)    # fix seed so results are consistent for the same ticker

    def _get_state(self, row):
        # 3 possible states based on which MA is higher
        diff = row['sma_short'] - row['sma_long']
        if diff > 0:
            return 'Above'
        elif diff < 0:
            return 'Below'
        else:
           return 'Equal'  # this case is pretty rare in practice

    def _choose_action(self, state):
        if np.random.rand() < self.epsilon or state not in self.q_table:
            return np.random.choice(['BUY', 'SELL', 'HOLD'])
        return max(self.q_table[state], key=self.q_table[state].get)

    def _ensure_state(self, state):
        # initialise Q values to 0 if we havent seen this state before
        if state not in self.q_table:
            self.q_table[state] = {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 0.0}

    def train(self, df: pd.DataFrame, episodes=10):
        df = df.copy()

        # loop through the training data multiple times
        # more episodes = more learning but also slower, 10 seemed enough in testing
        for _ in range(episodes):
            position = 0   # 0 = no shares held, 1 = holding shares
            buy_price = 0.0

            for i in range(len(df) - 1):
                row = df.iloc[i]
                next_row = df.iloc[i + 1]

                state = self._get_state(row)
                next_state = self._get_state(next_row)
                action = self._choose_action(state)

                reward = 0.0
                if action == 'BUY' and position == 0:
                    position = 1
                    buy_price = row['close']
                elif action == 'SELL' and position == 1:
                    reward = row['close'] - buy_price
                    position = 0
                elif action == 'HOLD':
                    reward = 0.01

                self._ensure_state(state)
                self._ensure_state(next_state)

                # bellman equation to update Q value
                # Q(s,a) = Q(s,a) + alpha * (reward + gamma * max_Q(s') - Q(s,a))
                max_future_q = max(self.q_table[next_state].values())
                self.q_table[state][action] += self.alpha * (
                    reward + self.gamma * max_future_q - self.q_table[state][action]
                )

        # after training, apply the greedy policy to get signals for each day
        # 1 = BUY, -1 = SELL, 0 = HOLD
        signals = []
        for i in range(len(df)):
            state = self._get_state(df.iloc[i])
            if state in self.q_table:
                best_action = max(self.q_table[state], key=self.q_table[state].get)
                signals.append(1 if best_action == 'BUY' else -1 if best_action == 'SELL' else 0)
            else:
                signals.append(0)  # default to hold if state not seen during training

        df['signal'] = signals
        return df
