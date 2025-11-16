from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS  # Import CORS
import pandas as pd
import numpy as np
import time
import threading
import logging
import sys
import ccxt
import yfinance as yf  # Import yfinance
from datetime import datetime, timedelta
import json
import os
import random

# Import from main.py with error handling
try:
    from main import TradingEnvironment, DQNTrader, preprocess_data
    print("Successfully imported from main.py")
except ImportError as e:
    print(f"Error importing from main.py: {e}")
    
    # Define fallback classes if import fails
    class DQNTrader:
        def __init__(self, state_size, feature_size, **kwargs):
            self.state_size = state_size
            self.feature_size = feature_size
            
        def act(self, state):
            # Random action: 0=hold, 1=buy, 2=sell
            return np.random.randint(0, 3)
            
        def load(self, filepath):
            print(f"Mock loading model from {filepath}")
            
    class TradingEnvironment:
        def __init__(self, data, state_size, **kwargs):
            pass
            
    def preprocess_data(df):
        return np.array([[0.5]])
        
    print("Using fallback classes for trading")
except Exception as e:
    print(f"Unexpected error importing from main.py: {e}")
    import traceback
    traceback.print_exc()

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables
trading_data = {
    'is_trading': False,
    'portfolio': {'USD': 10000000, 'BTC': 0},
    'trades': [],
    'last_action_time': None,
    'total_trades': 0,
    'successful_trades': 0,
    'failed_trades': 0,
    'initial_investment': 10000000,
    'start_time': None,
    'current_price': 0,
    'logs': [],
    'price_data_collected': False  # Flag to track if price data has been collected
}

price_history = []
model = None
trading_thread = None
stop_event = threading.Event()

def load_trained_model():
    """Load the trained DQN model"""
    global model
    try:
        # Initialize the model with the same parameters as in training
        state_size = 30
        feature_size = 12  # Match the model's expected feature size
        
        # Create model
        model = DQNTrader(state_size=state_size, feature_size=feature_size)
        
        # Load the model weights if file exists
        model_path = "dqn_trader_model.h5"
        if not os.path.exists(model_path):
            logger.warning(f"Model file {model_path} not found. Using untrained model.")
            print(f"Model file {model_path} not found. Using untrained model.")
            # Try alternative model file
            alternative_path = "best_model.h5"
            if os.path.exists(alternative_path):
                logger.info(f"Trying alternative model file: {alternative_path}")
                print(f"Trying alternative model file: {alternative_path}")
                model.load(alternative_path)
                logger.info(f"Successfully loaded model from {alternative_path}")
                print(f"Successfully loaded model from {alternative_path}")
        else:
            model.load(model_path)
            logger.info(f"Successfully loaded trained model from {model_path}")
            print(f"Successfully loaded trained model from {model_path}")
        
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        
        # Return a basic model that can still function
        logger.info("Creating a basic model that can still function")
        print("Creating a basic model that can still function")
        return DQNTrader(state_size=30, feature_size=12)

def calculate_technical_indicators(prices):
    """Calculate technical indicators from price history"""
    try:
        if len(prices) < 26:
            return None
        
        # Calculate RSI
        deltas = np.diff(prices)
        seed = deltas[:14]
        up = seed[seed >= 0].sum() / 14.0
        down = -seed[seed < 0].sum() / 14.0
        if down != 0:
            rs = up / down
        else:
            rs = 1.0
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        # Calculate MACD
        ema12 = np.mean(prices[-12:])
        ema26 = np.mean(prices)
        macd = ema12 - ema26
        
        return {
            'rsi': float(rsi),
            'macd': float(macd)
        }
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {e}")
        return None

def calculate_portfolio_stats():
    """Calculate portfolio statistics"""
    try:
        if not price_history or trading_data['current_price'] == 0:
            logger.warning("No price history or current price is 0, returning default stats")
            return {
                'portfolio_value': trading_data['portfolio']['USD'],
                'profit_loss': 0,
                'profit_loss_percent': 0,
                'total_trades': trading_data['total_trades'],
                'successful_trades': trading_data['successful_trades'],
                'failed_trades': trading_data['failed_trades'],
                'success_rate': 0
            }
        
        # Get the current price and calculate portfolio value
        current_price = float(trading_data['current_price'])
        btc_value = trading_data['portfolio']['BTC'] * current_price
        usd_value = trading_data['portfolio']['USD']
        portfolio_value = usd_value + btc_value
        
        # Calculate profit/loss based on initial investment
        initial_value = float(trading_data['initial_investment'])
        profit_loss = portfolio_value - initial_value
        
        # Calculate profit/loss percentage
        profit_loss_percent = (profit_loss / initial_value) * 100 if initial_value > 0 else 0
        
        # Calculate success rate
        success_rate = (trading_data['successful_trades'] / trading_data['total_trades'] * 100) if trading_data['total_trades'] > 0 else 0
        
        # Log the calculation for debugging
        logger.info(f"Portfolio stats: Value={portfolio_value:.2f}, Initial={initial_value:.2f}, P/L={profit_loss:.2f}, P/L%={profit_loss_percent:.2f}%")
        
        # Only log to UI occasionally to avoid spam
        if random.random() < 0.1:  # Log approximately 10% of the time
            add_log(f"Portfolio value: {portfolio_value:.2f}, P/L: {profit_loss:.2f} ({profit_loss_percent:.2f}%)")
        
        # Create stats object with explicit type conversion
        stats = {
            'portfolio_value': float(round(portfolio_value, 2)),
            'profit_loss': float(round(profit_loss, 2)),
            'profit_loss_percent': float(round(profit_loss_percent, 2)),
            'total_trades': int(trading_data['total_trades']),
            'successful_trades': int(trading_data['successful_trades']),
            'failed_trades': int(trading_data['failed_trades']),
            'success_rate': float(round(success_rate, 2))
        }
        
        # Debug log the stats object
        logger.debug(f"Returning stats: {stats}")
        
        return stats
    except Exception as e:
        logger.error(f"Error calculating portfolio stats: {e}")
        import traceback
        traceback.print_exc()
        return {
            'portfolio_value': trading_data['portfolio']['USD'],
            'profit_loss': 0,
            'profit_loss_percent': 0,
            'total_trades': trading_data['total_trades'],
            'successful_trades': trading_data['successful_trades'],
            'failed_trades': trading_data['failed_trades'],
            'success_rate': 0
        }

def prepare_state_for_model(current_price):
    """Prepare the current state for model prediction"""
    try:
        if len(price_history) < 30:
            return None
        
        # Get the last 30 prices
        prices = price_history[-30:]
        
        # Calculate technical indicators
        indicators = calculate_technical_indicators(prices)
        if not indicators:
            return None
        
        # Create state array
        price_array = np.array(prices).reshape(-1, 1)
        
        # Normalize prices (simple min-max scaling)
        min_price = np.min(price_array)
        max_price = np.max(price_array)
        if max_price > min_price:
            normalized_prices = (price_array - min_price) / (max_price - min_price)
        else:
            normalized_prices = price_array - min_price
        
        # Create additional features
        rsi = indicators['rsi']
        macd = indicators['macd']
        position_size = trading_data['portfolio']['BTC']
        avg_buy_price = sum([trade['price'] for trade in trading_data['trades'] if trade['action'] == 'buy']) / max(1, len([t for t in trading_data['trades'] if t['action'] == 'buy']))
        normalized_cash = trading_data['portfolio']['USD'] / max(1, trading_data['initial_investment'])
        position_flag = 1 if position_size > 0 else 0
        
        # Add an additional feature to match the model's expected 12 features
        time_feature = np.linspace(0, 1, 30)  # Time feature from 0 to 1
        
        # Create state with additional features (now 7 additional features for a total of 12)
        extra_features = np.column_stack((
            np.full(30, rsi),
            np.full(30, macd),
            np.full(30, normalized_cash),
            np.full(30, position_size),
            np.full(30, avg_buy_price),
            np.full(30, position_flag),
            time_feature  # Additional feature to match the model
        ))
        
        # Combine price and features
        state = np.hstack([normalized_prices, extra_features])
        
        # Add batch dimension
        return state[np.newaxis, :, :]
    except Exception as e:
        logger.error(f"Error preparing state: {e}")
        return None

# Helper function to add a log message
def add_log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    log_entry = {
        'timestamp': timestamp,
        'message': message
    }
    trading_data['logs'].append(log_entry)
    print(f"[{timestamp}] {message}")
    logger.info(message)
    
    # Keep only the last 50 log messages
    if len(trading_data['logs']) > 50:
        trading_data['logs'].pop(0)

def get_real_time_price(symbol='BTC-USD'):
    """Get real-time cryptocurrency price from Yahoo Finance"""
    try:
        # Use yfinance to get the latest price
        ticker = yf.Ticker(symbol)
        data = ticker.history(period='1d', interval='1m')
        
        if not data.empty:
            # Get the latest closing price
            latest_price = data['Close'].iloc[-1]
            add_log(f"Latest {symbol} price from Yahoo Finance: ${latest_price:.2f}")
            return float(latest_price)
        else:
            # Fallback to random price if no data
            last_price = price_history[-1] if price_history else 83000
            fallback_price = last_price + np.random.normal(0, 100)
            add_log(f"No data from Yahoo Finance, using fallback price: ${fallback_price:.2f}")
            return fallback_price
    except Exception as e:
        logger.error(f"Error fetching price from Yahoo Finance: {e}")
        print(f"Error fetching price from Yahoo Finance: {e}")
        
        # Try CCXT as a backup
        try:
            exchange = ccxt.binance()
            ticker = exchange.fetch_ticker('BTC/USDT')
            price = float(ticker['last'])
            add_log(f"Fallback to CCXT price: ${price:.2f}")
            return price
        except Exception as ccxt_error:
            logger.error(f"Error fetching price from CCXT: {ccxt_error}")
            print(f"Error fetching price from CCXT: {ccxt_error}")
            
            # Final fallback to random price
            if price_history:
                last_price = price_history[-1]
                # Add some small random movement
                random_change = last_price * np.random.uniform(-0.001, 0.001)
                new_price = last_price + random_change
                add_log(f"Using simulated price: ${new_price:.2f}")
                return new_price
            else:
                # If no price history, use a default price with some randomness
                default_price = 83000 + np.random.normal(0, 100)
                add_log(f"Using default price: ${default_price:.2f}")
                return default_price

def execute_trade(action, price, amount):
    """Execute a trade based on model prediction"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # Convert string action to numeric if needed
        if isinstance(action, str):
            action = 1 if action.lower() == 'buy' else 2 if action.lower() == 'sell' else 0
        
        if action == 1:  # Buy
            cost = price * amount
            if trading_data['portfolio']['USD'] >= cost:
                # Update portfolio
                trading_data['portfolio']['USD'] -= cost
                trading_data['portfolio']['BTC'] += amount
                
                # Calculate total and portfolio value
                total = cost
                portfolio_value = trading_data['portfolio']['USD'] + trading_data['portfolio']['BTC'] * price
                
                # Create trade record
                trade = {
                    'action': 'buy',
                    'price': float(price),
                    'amount': float(amount),
                    'timestamp': timestamp,
                    'total': float(total),
                    'portfolio_value': float(portfolio_value)
                }
                
                # Update trading data
                trading_data['trades'].append(trade)
                trading_data['last_action_time'] = datetime.now()
                trading_data['total_trades'] += 1
                trading_data['successful_trades'] += 1
                
                # Log the trade
                add_log(f"BUY executed: {amount:.8f} BTC at ${price:.2f} for ${total:.2f}")
                
                # Update portfolio stats after trade
                stats = calculate_portfolio_stats()
                add_log(f"Portfolio after BUY: USD={trading_data['portfolio']['USD']:.2f}, BTC={trading_data['portfolio']['BTC']:.8f}, P/L={stats['profit_loss']:.2f}")
                
                return True
            else:
                add_log(f"BUY failed: Not enough USD. Required: ${cost:.2f}, Available: ${trading_data['portfolio']['USD']:.2f}")
                trading_data['total_trades'] += 1
                trading_data['failed_trades'] += 1
                return False
                
        elif action == 2:  # Sell
            if trading_data['portfolio']['BTC'] >= amount:
                # Update portfolio
                revenue = price * amount
                trading_data['portfolio']['USD'] += revenue
                trading_data['portfolio']['BTC'] -= amount
                
                # Calculate total and portfolio value
                total = revenue
                portfolio_value = trading_data['portfolio']['USD'] + trading_data['portfolio']['BTC'] * price
                
                # Create trade record
                trade = {
                    'action': 'sell',
                    'price': float(price),
                    'amount': float(amount),
                    'timestamp': timestamp,
                    'total': float(total),
                    'portfolio_value': float(portfolio_value)
                }
                
                # Update trading data
                trading_data['trades'].append(trade)
                trading_data['last_action_time'] = datetime.now()
                trading_data['total_trades'] += 1
                trading_data['successful_trades'] += 1
                
                # Log the trade
                add_log(f"SELL executed: {amount:.8f} BTC at ${price:.2f} for ${total:.2f}")
                
                # Update portfolio stats after trade
                stats = calculate_portfolio_stats()
                add_log(f"Portfolio after SELL: USD={trading_data['portfolio']['USD']:.2f}, BTC={trading_data['portfolio']['BTC']:.8f}, P/L={stats['profit_loss']:.2f}")
                
                return True
            else:
                add_log(f"SELL failed: Not enough BTC. Required: {amount:.8f}, Available: {trading_data['portfolio']['BTC']:.8f}")
                trading_data['total_trades'] += 1
                trading_data['failed_trades'] += 1
                return False
        else:
            # Hold action
            portfolio_value = trading_data['portfolio']['USD'] + trading_data['portfolio']['BTC'] * price
            
            # Create trade record for hold action
            trade = {
                'action': 'hold',
                'price': float(price),
                'amount': 0,
                'timestamp': timestamp,
                'total': 0,
                'portfolio_value': float(portfolio_value)
            }
            
            # Update trading data
            trading_data['trades'].append(trade)
            trading_data['last_action_time'] = datetime.now()
            trading_data['total_trades'] += 1
            trading_data['successful_trades'] += 1
            
            # Log the hold action
            add_log(f"HOLD executed at price ${price:.2f}")
            
            # Update portfolio stats for hold action
            stats = calculate_portfolio_stats()
            add_log(f"Portfolio during HOLD: USD={trading_data['portfolio']['USD']:.2f}, BTC={trading_data['portfolio']['BTC']:.8f}, P/L={stats['profit_loss']:.2f}")
            
            return True
            
    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        add_log(f"Trade execution error: {str(e)}")
        return False

def trading_loop():
    """Background thread for continuous trading"""
    add_log("Trading loop started")
    
    # Keep only the last 5 minutes of price data (assuming 1-second intervals)
    max_price_history = 300
    
    # Flag to track if we need to collect initial data
    need_initial_data = not trading_data['price_data_collected'] or len(price_history) < 30
    
    if need_initial_data:
        add_log("Starting to collect initial price data")
    else:
        add_log("Using existing price data, ready to trade")
    
    while not stop_event.is_set():
        try:
            # Get current price from Yahoo Finance
            current_price = get_real_time_price('BTC-USD')
            trading_data['current_price'] = current_price
            
            # Add to price history
            price_history.append(current_price)
            
            # Limit price history size
            if len(price_history) > max_price_history:
                price_history.pop(0)
            
            # Calculate portfolio stats to update profit/loss
            calculate_portfolio_stats()
            
            # Check if we have enough data
            if need_initial_data and len(price_history) < 30:
                add_log(f"Collecting price data: {len(price_history)}/30")
                time.sleep(5)  # Longer sleep when collecting initial data
                continue
            elif need_initial_data and len(price_history) >= 30:
                add_log("Initial price data collection complete, ready to trade")
                trading_data['price_data_collected'] = True
                need_initial_data = False
            
            # Prepare state for model
            state = prepare_state_for_model(current_price)
            if state is None:
                add_log("Failed to prepare state for model, skipping prediction")
                time.sleep(1)
                continue
            
            # Get model prediction
            try:
                if model:
                    state_reshaped = np.expand_dims(state, axis=0)
                    action = model.act(state_reshaped)
                    
                    # Map action to trade
                    # 0 = hold, 1 = buy, 2 = sell
                    if action == 1:  # Buy
                        # Calculate amount to buy (10% of available USD)
                        usd_available = trading_data['portfolio']['USD']
                        amount_to_buy = (usd_available * 0.1) / current_price
                        
                        if amount_to_buy > 0:
                            execute_trade('buy', current_price, amount_to_buy)
                        else:
                            add_log("Not enough USD to buy")
                    
                    elif action == 2:  # Sell
                        # Calculate amount to sell (10% of available BTC)
                        btc_available = trading_data['portfolio']['BTC']
                        amount_to_sell = btc_available * 0.1
                        
                        if amount_to_sell > 0:
                            execute_trade('sell', current_price, amount_to_sell)
                        else:
                            add_log("Not enough BTC to sell")
                    else:  # Hold (action == 0)
                        execute_trade('hold', current_price, 0)
                        add_log(f"HOLD action taken at price ${current_price:.2f}")
                        # Update portfolio stats for hold action
                        stats = calculate_portfolio_stats()
                        add_log(f"Portfolio after HOLD: USD={trading_data['portfolio']['USD']:.2f}, BTC={trading_data['portfolio']['BTC']:.8f}, P/L={stats['profit_loss']:.2f}")
                else:
                    add_log("Model not loaded, skipping prediction")
            except Exception as model_error:
                logger.error(f"Error in model prediction: {model_error}")
                add_log(f"Error in model prediction: {str(model_error)}")
            
            # Sleep for a short time before next iteration
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
            add_log(f"Error in trading loop: {str(e)}")
            time.sleep(5)  # Sleep longer on error
    
    add_log("Trading loop stopped")

# Serve static files
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/start_trading', methods=['POST'])
def start_trading():
    try:
        print("Received start_trading request")
        data = request.get_json()
        print(f"Request data: {data}")
        
        initial_investment = float(data.get('investment', 10000000))
        print(f"Initial investment: {initial_investment}")
        
        # Reset portfolio and trading stats, but keep price history
        trading_data['portfolio']['USD'] = initial_investment
        trading_data['portfolio']['BTC'] = 0
        trading_data['trades'] = []
        trading_data['last_action_time'] = None
        trading_data['total_trades'] = 0
        trading_data['successful_trades'] = 0
        trading_data['failed_trades'] = 0
        trading_data['initial_investment'] = initial_investment
        trading_data['start_time'] = datetime.now()
        
        # Only clear price history if it's empty or if explicitly requested
        reset_price_history = data.get('reset_price_history', False)
        if reset_price_history or len(price_history) == 0:
            price_history.clear()
            trading_data['price_data_collected'] = False
            add_log("Price history reset, will collect new data")
        
        if not trading_data['is_trading']:
            global model, trading_thread, stop_event
            
            # Load model if not already loaded
            if model is None:
                print("Loading model...")
                try:
                    model = load_trained_model()
                    print("Model loaded successfully")
                except Exception as e:
                    print(f"Error loading model: {e}")
                    raise
            
            # Reset stop event
            stop_event.clear()
            
            # Start trading thread
            print("Starting trading thread...")
            trading_thread = threading.Thread(target=trading_loop)
            trading_thread.daemon = True
            trading_thread.start()
            
            trading_data['is_trading'] = True
            print("Trading started successfully")
            
            return jsonify({
                'status': 'success',
                'message': 'Trading started',
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            })
        else:
            print("Trading already in progress")
            return jsonify({
                'status': 'error',
                'message': 'Trading already in progress',
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            })
    except Exception as e:
        logger.error(f"Error starting trading: {e}")
        print(f"Error starting trading: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        }), 500

@app.route('/stop_trading', methods=['POST'])
def stop_trading():
    try:
        global stop_event, trading_thread
        
        if trading_data['is_trading']:
            # Set stop event to signal thread to stop
            stop_event.set()
            
            # Wait for thread to finish
            if trading_thread and trading_thread.is_alive():
                trading_thread.join(timeout=5)
            
            trading_data['is_trading'] = False
            
            # Calculate final stats
            stats = calculate_portfolio_stats()
            
            return jsonify({
                'status': 'success',
                'message': 'Trading stopped',
                'stats': stats,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'No trading in progress',
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            })
    except Exception as e:
        logger.error(f"Error stopping trading: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        }), 500

@app.route('/get_status', methods=['GET'])
def get_status():
    try:
        # Get current price
        current_price = trading_data['current_price']
        
        # Calculate portfolio stats
        stats = calculate_portfolio_stats()
        
        # Get recent trades (last 20)
        recent_trades = trading_data['trades'][-20:] if trading_data['trades'] else []
        
        # Get recent prices (last 100 for chart)
        recent_prices = []
        if price_history:
            for i, p in enumerate(price_history[-100:]):
                timestamp = (datetime.now() - timedelta(seconds=len(price_history[-100:])-i-1)).strftime("%H:%M:%S")
                
                # Find any trades that occurred at this timestamp
                trade_actions = []
                for trade in trading_data['trades']:
                    trade_time = datetime.strptime(trade['timestamp'], "%Y-%m-%d %H:%M:%S.%f").strftime("%H:%M:%S")
                    if trade_time == timestamp and trade['action'] in ['buy', 'sell']:
                        trade_actions.append({
                            'action': trade['action'],
                            'price': float(trade['price']),
                            'amount': float(trade['amount']),
                            'portfolio_value': float(trade['portfolio_value'])
                        })
                
                recent_prices.append({
                    'price': float(p),
                    'timestamp': timestamp,
                    'trades': trade_actions
                })
            
            # If no price history, add at least the current price
            if not recent_prices and current_price:
                recent_prices.append({
                    'price': float(current_price),
                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                    'trades': []
                })
        
        # Get recent logs (last 20)
        recent_logs = trading_data['logs'][-20:] if trading_data['logs'] else []
        
        # Log the response data for debugging
        logger.debug(f"Sending status: price={current_price}, portfolio={trading_data['portfolio']}, stats={stats}")
        
        return jsonify({
            'status': 'success',
            'is_trading': trading_data['is_trading'],
            'current_price': float(current_price),
            'portfolio': {
                'USD': float(trading_data['portfolio']['USD']),
                'BTC': float(trading_data['portfolio']['BTC'])
            },
            'stats': stats,
            'trades': recent_trades,
            'prices': recent_prices,
            'logs': recent_logs,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        })
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        }), 500

if __name__ == '__main__':
    app.run(debug=True, threaded=True) 