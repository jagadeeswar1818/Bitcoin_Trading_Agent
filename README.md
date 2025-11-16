# AI Bitcoin Trading Bot

A real-time Bitcoin trading application powered by Deep Q-Learning (DQN) that simulates trading strategies with live price data.

![Bitcoin Trading Bot]
## Features

- **Real-time Price Data**: Fetches live Bitcoin prices from Yahoo Finance
- **Deep Q-Learning Agent**: Uses a DQN model to make trading decisions
- **Interactive Dashboard**: Dark-themed UI with real-time price chart and trade markers
- **Portfolio Tracking**: Monitors portfolio value, profit/loss, and trading statistics
- **Trade History**: Displays a detailed history of all executed trades
- **Trading Logs**: Shows real-time logs of trading activities and system messages

## Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow 2.15.0
- CUDA 12.x (for GPU acceleration)
- Flask
- Other dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/bitcoin-trading-bot.git
   cd bitcoin-trading-bot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up CUDA environment variables (for GPU acceleration):
   ```bash
   # Windows
   set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
   set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
   set CUDA_PATH_V12_8=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
   
   # Linux/macOS
   export CUDA_HOME=/usr/local/cuda-12.8
   export CUDA_PATH=/usr/local/cuda-12.8
   export CUDA_PATH_V12_8=/usr/local/cuda-12.8
   ```

### Running the Application

1. Start the Flask server:
   ```bash
   python app.py
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

## Usage

1. **Initial Setup**:
   - Enter your initial investment amount
   - Optionally check "Reset Price History" to collect new price data
   - Click "Start Trading" to begin

2. **Trading Controls**:
   - Use the "Start Trading" button to begin a trading session
   - Use the "Stop Trading" button to end the current session
   - You can restart trading with a different investment amount without losing price data

3. **Monitoring**:
   - The price chart shows real-time Bitcoin prices with buy/sell markers
   - Portfolio statistics display current value and profit/loss
   - Trading logs show system messages and trade executions
   - Trade history table shows all executed trades

## How It Works

### DQN Trading Agent

The application uses a Deep Q-Network (DQN) reinforcement learning agent to make trading decisions:

- **State**: Combination of price data, technical indicators, and portfolio status
- **Actions**: Buy, Sell, or Hold Bitcoin
- **Reward**: Change in portfolio value after each action

The agent is trained to maximize long-term portfolio value by learning optimal trading strategies.

### Technical Architecture

- **Backend**: Flask server with Python for data processing and model inference
- **Frontend**: HTML, CSS, and JavaScript for the user interface
- **Data Source**: Yahoo Finance API for real-time Bitcoin prices
- **Model**: TensorFlow implementation of DQN with LSTM layers

## Customization

You can customize various aspects of the trading bot:

- **Model Parameters**: Modify the DQN architecture in `main.py`
- **Trading Strategy**: Adjust the reward function and action selection in `app.py`
- **UI Theme**: Modify the dark theme in `static/style.css`
- **Technical Indicators**: Add or remove indicators in the `calculate_technical_indicators` function

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TensorFlow team for the deep learning framework
- Chart.js for the interactive charting library
- Yahoo Finance for providing real-time price data

## Disclaimer

This application is for educational purposes only. Trading cryptocurrencies involves significant risk. Do not use this system for actual trading without thorough testing and risk management strategies. 