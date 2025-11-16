import numpy as np
import pandas as pd
import random
from collections import deque
from keras.models import Sequential, clone_model, load_model, save_model
from keras.layers import Dense, LSTM, Dropout, BatchNormalization
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from typing import Tuple, List, Union
import tensorflow as tf
import os

# -------------------------------
# Environment definition
# -------------------------------
class TradingEnvironment:
    def __init__(self, data: np.ndarray, state_size: int, transaction_cost: float = 0.001):
        """
        data: np.ndarray with each row representing a timestep and columns for features.
        state_size: number of timesteps to use as the state.
        transaction_cost: cost percentage per trade.
        """
        self.data = data
        self.state_size = state_size
        # The feature size is the original data features plus 6 extra features
        self.feature_size = data.shape[1] + 6
        self.transaction_cost = transaction_cost
        self.reset()

    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.inventory = []
        self.total_profit = 0
        self.cash = 10000  # Starting cash amount
        self.portfolio_value_history = [self.cash]
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        # Get a window of price data and original features
        price_data = self.data[self.current_step:self.current_step + self.state_size]

        # Calculate additional features
        rsi = self._calculate_rsi(14)
        macd = self._calculate_macd()

        # Create position and portfolio features
        position_size = len(self.inventory)
        avg_position_price = np.mean(self.inventory) if self.inventory else 0
        normalized_cash = self.cash / 10000

        # Create additional features arrays (repeat each value for the length of the state window)
        additional_features = np.column_stack((
            np.full(self.state_size, rsi),
            np.full(self.state_size, macd),
            np.full(self.state_size, normalized_cash),
            np.full(self.state_size, position_size),
            np.full(self.state_size, avg_position_price),
            np.full(self.state_size, 1 if position_size > 0 else 0)  # Position flag
        ))

        # Combine the raw data with additional features
        state = np.column_stack((price_data, additional_features))
        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Actions:
          0: Hold (do nothing)
          1: Buy
          2: Sell
        """
        current_price = self.data[self.current_step][0]  # assuming first column is Price
        done = False
        reward = 0

        # Execute the selected trading action
        if action == 1:  # Buy
            if self.cash >= current_price * (1 + self.transaction_cost):
                # Buy at most 1 share at a time
                max_shares = self.cash // (current_price * (1 + self.transaction_cost))
                shares_to_buy = min(1, max_shares)
                cost = current_price * shares_to_buy * (1 + self.transaction_cost)
                self.cash -= cost
                self.inventory.extend([current_price] * int(shares_to_buy))
                reward = -self.transaction_cost * current_price

        elif action == 2:  # Sell
            if len(self.inventory) > 0:
                bought_price = self.inventory.pop(0)
                sale_proceed = current_price * (1 - self.transaction_cost)
                self.cash += sale_proceed
                profit = sale_proceed - bought_price
                reward = profit

        # Update portfolio value and track history
        portfolio_value = self.cash + len(self.inventory) * current_price
        self.portfolio_value_history.append(portfolio_value)

        # Add a Sharpe ratio–based component to the reward (optional risk-adjusted measure)
        if len(self.portfolio_value_history) > 1:
            returns = np.diff(self.portfolio_value_history[-252:]) / self.portfolio_value_history[-252:-1]
            if returns.std() != 0:
                sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std())
            else:
                sharpe_ratio = 0
            reward += sharpe_ratio * 0.1

        self.current_step += 1

        # Check if we have reached the end of the data
        if self.current_step + self.state_size >= len(self.data):
            done = True
            final_return = (portfolio_value - 10000) / 10000
            reward += final_return * 10  # Bonus reward for overall return

        info = {
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'inventory': len(self.inventory),
            'current_price': current_price
        }

        next_state = self._get_state()
        return next_state, reward, done, info

    def _calculate_rsi(self, period: int) -> float:
        if self.current_step < period:
            return 50  # neutral value when not enough data
        prices = self.data[self.current_step - period:self.current_step, 0]
        deltas = np.diff(prices)
        gain = np.clip(deltas, 0, None).mean()
        loss = -np.clip(deltas, None, 0).mean()

        if loss == 0:
            return 100
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self) -> float:
        if self.current_step < 26:
            return 0
        prices = self.data[self.current_step - 26:self.current_step, 0]
        # Calculate exponential moving averages (EMAs)
        ema12 = np.exp(np.linspace(-1, 0, 12)) @ prices[-12:] / np.exp(np.linspace(-1, 0, 12)).sum()
        ema26 = np.exp(np.linspace(-1, 0, 26)) @ prices / np.exp(np.linspace(-1, 0, 26)).sum()
        return ema12 - ema26

# -------------------------------
# RL Agent (DQN) definition
# -------------------------------
class DQNTrader:
    def __init__(self, state_size: int, feature_size: int, action_space: int = 3):
        self.state_size = state_size
        self.feature_size = feature_size
        self.action_space = action_space

        self.memory = deque(maxlen=10000)
        self.gamma = 0.95             # Discount factor for future rewards
        self.epsilon = 1.0            # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32

        self.model = self._build_model()
        self.target_model = clone_model(self.model)
        self.update_target_model()

    def _build_model(self) -> Sequential:
        # Build a model using LSTM layers so that the agent can capture temporal relationships
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(self.state_size, self.feature_size)),
            BatchNormalization(),
            Dropout(0.3),
            LSTM(100, return_sequences=False),
            BatchNormalization(),
            Dropout(0.3),
            Dense(100, activation='relu'),
            BatchNormalization(),
            Dense(self.action_space, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        # Copy weights from the main model to the target model
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state: np.ndarray) -> int:
        # Choose an action using epsilon-greedy strategy
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        action_values = self.model.predict(state, verbose=0)
        return np.argmax(action_values[0])

    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size: int):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states = np.array([x[0][0] for x in minibatch])
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch])
        next_states = np.array([x[3][0] for x in minibatch])
        dones = np.array([x[4] for x in minibatch])

        # Calculate the target Q values
        target_q = rewards + self.gamma * np.max(self.target_model.predict(next_states, verbose=0), axis=1) * (1 - dones)
        target_f = self.model.predict(states, verbose=0)
        target_f[np.arange(batch_size), actions] = target_q

        self.model.fit(states, target_f, epochs=1, verbose=0, batch_size=batch_size)

        # Reduce exploration as training progresses
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, filepath: str):
        """Load model weights from a file"""
        try:
            self.model = load_model(filepath)
            self.target_model = clone_model(self.model)
            self.update_target_model()
            print(f"Successfully loaded model from {filepath}")
        except Exception as e:
            print(f"Error loading model from {filepath}: {e}")

    def save(self, filepath: str):
        """Save model weights to a file"""
        try:
            self.model.save(filepath)
            print(f"Successfully saved model to {filepath}")
        except Exception as e:
            print(f"Error saving model to {filepath}: {e}")

# -------------------------------
# Data Preprocessing Function
# -------------------------------
def preprocess_data(data: pd.DataFrame) -> np.ndarray:
    # Helper function to convert volume strings like '1.2K', '3.4M' into float values
    def convert_volume(volume: Union[str, float]) -> float:
        if isinstance(volume, str):
            if 'K' in volume:
                return float(volume.replace('K', '')) * 1e3
            elif 'M' in volume:
                return float(volume.replace('M', '')) * 1e6
        return float(volume)

    # Replace any missing values represented by '-' with NaN
    data.replace('-', np.nan, inplace=True)
    for column in ['Price', 'Open', 'High', 'Low', 'Vol.', 'Change %']:
        if column == 'Vol.':
            data[column] = data[column].apply(convert_volume)
        else:
            data[column] = data[column].str.replace(',', '').str.replace('%', '').astype(float)

    # Calculate some technical indicators for additional features
    data['RSI'] = data['Price'].diff().rolling(window=14).apply(
        lambda x: 100 - (100 / (1 + (x[x > 0].mean() / -x[x < 0].mean()))) if (-x[x < 0].mean()) != 0 else 50)
    data['MACD'] = data['Price'].ewm(span=12).mean() - data['Price'].ewm(span=26).mean()
    data['Signal'] = data['MACD'].ewm(span=9).mean()
    data['BB_up'] = data['Price'].rolling(window=20).mean() + 2 * data['Price'].rolling(window=20).std()
    data['BB_down'] = data['Price'].rolling(window=20).mean() - 2 * data['Price'].rolling(window=20).std()

    features = ['Price', 'RSI', 'MACD', 'Signal', 'BB_up', 'BB_down']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features].dropna())

    return scaled_data

# -------------------------------
# Training and Evaluation Functions
# -------------------------------
def evaluate_agent(agent: DQNTrader, env: TradingEnvironment) -> float:
    state = env.reset()
    total_reward = 0 
    done = False

    while not done:
        state_reshaped = np.expand_dims(state, axis=0)
        action = agent.act(state_reshaped)
        next_state, reward, done, info = env.step(action)
        state = next_state
        total_reward += reward

    return total_reward

def train_dqn_agent(data: np.ndarray, state_size: int, episodes: int,
                    validation_split: float = 0.2) -> Tuple[DQNTrader, List[float]]:
    # Split data into training and validation sets
    train_size = int(len(data) * (1 - validation_split))
    train_data = data[:train_size]
    val_data = data[train_size:]

    env = TradingEnvironment(train_data, state_size)
    agent = DQNTrader(state_size, env.feature_size)
    rewards = []
    val_rewards = []
    best_val_reward = float('-inf')
    best_weights_path = 'best_model.weights.h5'

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            state_reshaped = np.expand_dims(state, axis=0)
            action = agent.act(state_reshaped)
            next_state, reward, done, info = env.step(action)
            agent.remember(state_reshaped, action, reward,
                           np.expand_dims(next_state, axis=0), done)
            state = next_state
            total_reward += reward

            if len(agent.memory) >= agent.batch_size:
                agent.replay(agent.batch_size)

        rewards.append(total_reward)

        # Periodically evaluate on validation set and save the best performing model
        if episode % 5 == 0:
            val_env = TradingEnvironment(val_data, state_size)
            val_reward = evaluate_agent(agent, val_env)
            val_rewards.append(val_reward)

            if val_reward > best_val_reward:
                best_val_reward = val_reward
                # Save only the weights (you can also save the full model if preferred)
                agent.model.save_weights(best_weights_path)

            print(f"Episode {episode + 1}/{episodes}")
            print(f"Training Reward: {total_reward:.2f}")
            print(f"Validation Reward: {val_reward:.2f}")
            print(f"Epsilon: {agent.epsilon:.2f}")
            print("--------------------")

        if episode % 10 == 0:
            agent.update_target_model()

    # After training, load the best model weights
    agent.model.load_weights(best_weights_path)
    return agent, rewards

# -------------------------------
# Main Training Script
# -------------------------------
if __name__ == "__main__":
    # Make sure to update the file path to your Bitcoin historical data CSV file
    data_filepath = "D:\Downloads\Bitcoin Historical Data 1.csv"
    if not os.path.exists(data_filepath):
        raise FileNotFoundError(f"CSV file not found at: {data_filepath}")

    bitcoin_data = pd.read_csv(data_filepath)
    processed_data = preprocess_data(bitcoin_data)

    state_size = 30   # Number of timesteps used as state input
    episodes = 50     # Number of training episodes

    # Train the agent
    agent, rewards = train_dqn_agent(processed_data, state_size, episodes)

    # Optionally, save the complete model architecture and weights for deployment
    deployment_model_path = "dqn_trader_model.h5"
    agent.model.save(deployment_model_path)
    print(f"Model saved for deployment at: {deployment_model_path}")

    # Plot training rewards over episodes
    plt.figure(figsize=(12, 6))
    plt.plot(rewards)
    plt.title('Training Rewards Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()