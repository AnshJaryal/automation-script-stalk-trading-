📈 AI-Driven Algorithmic Trading with Deep Q-Networks (DQN)

A complete end-to-end pipeline for automated stock trading using Deep Reinforcement Learning. This project features a custom trading environment, automated technical indicator engineering, and a live trading bot integrated with the Alpaca Trade API.
🛠️ Project Architecture

The system is divided into three core modules:
1. Feature Engineering & Technical Analysis

The bot doesn't just look at raw prices; it uses pandas-ta to compute sophisticated market signals.

    Indicators: RSI (Relative Strength Index), SMA (Simple Moving Averages), and MACD (Moving Average Convergence Divergence).

    Preprocessing: Automated cleaning and normalization of historical CSV data (e.g., TSLA, AAPL).

2. Deep Q-Network (DQN) Agent

The "brain" of the bot is a Reinforcement Learning agent built with PyTorch.

    Neural Network: A multi-layer perceptron (MLP) that maps market states to trading actions.

    Experience Replay: Uses a deque memory buffer to break correlation in training data and improve stability.

    Epsilon-Greedy Strategy: Implements an exploration/exploitation balance, starting with random trades and gradually shifting to learned optimal strategies.

    Target Network: Implements a dual-network strategy (Policy and Target) to stabilize the Q-value targets during training.

3. Live Execution Engine

A real-time trading script that bridges the gap between the model and the stock market.

    Broker Integration: Uses Alpaca Trade API for paper trading and live market data.

    Inference Loop: Fetches 1-minute interval bars, processes the state, and executes Buy, Sell, or Hold orders automatically.

    Secure Config: Utilizes python-dotenv for secure API key management.
