import re
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# basic logging setup - INFO level so we can see whats happening without too much noise
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@app.route("/advice", methods=["POST"])
def get_advice():
    from data.fetch_stock_data import get_stock_data
    from strategies.basic_strategy import apply_moving_average_strategy
    from strategies.q_learning_strategy import QLearningTrader
    from engine.advisor import generate_advice
    from data.ticker_lookup import resolve_ticker

    data = request.get_json()
    raw_input = data.get("symbol", "").strip()

    log.info(f"resolve_ticker('{raw_input}') -> {resolve_ticker(raw_input)}")

    # resolve input to a ticker - handles company names, partial names, and direct tickers
    symbol = resolve_ticker(raw_input)

    if not symbol:
        return jsonify({"message": f"I wasn't able to find a ticker for '{raw_input}'. Please enter a valid stock symbol like AAPL or TSLA."}), 400

    # yfinance uses dots not dashes, so BRK-B becomes BRK.B
    symbol = symbol.replace("-", ".")

    try:
        stock_df = get_stock_data(symbol, period="6mo")

        if stock_df.empty or 'close' not in stock_df.columns:
            return jsonify({"message": f"No valid stock data found for {symbol}."}), 404

        # run the MA strategy first, then let the RL agent refine the signals
        stock_df = apply_moving_average_strategy(stock_df)

        q_bot = QLearningTrader()
        stock_df = q_bot.train(stock_df, episodes=50)

        if not stock_df.empty and 'signal' in stock_df.columns:
            advice = generate_advice(stock_df, symbol=symbol)
            return jsonify({"message": advice})
        else:
            return jsonify({"message": f"No signal data available for {symbol}."}), 204

    except Exception as e:
        log.error(f"advice failed for {symbol}: {e}")
        return jsonify({"message": f"Error generating advice for {symbol}."}), 500


@app.route("/evaluate", methods=["POST"])
def evaluate_strategy():
    from data.fetch_stock_data import get_stock_data
    from strategies.basic_strategy import apply_moving_average_strategy
    from strategies.q_learning_strategy import QLearningTrader
    from evaluation.backtester import backtest_strategy
    from data.ticker_lookup import resolve_ticker

    data = request.get_json()
    raw_input = data.get("symbol", "").strip()

    symbol = resolve_ticker(raw_input)

    if not symbol:
        return jsonify({"message": f"I wasn't able to find a ticker for '{raw_input}'. Please enter a valid stock symbol like AAPL or TSLA."}), 400

    try:
        # use 2 years of data for evaluation so we get a decent backtest window
        stock_df = get_stock_data(symbol, period="2y")

        stock_df = apply_moving_average_strategy(stock_df)

        if stock_df.empty or 'signal' not in stock_df.columns:
            return jsonify({"message": "No signal data available."}), 204

        q_bot = QLearningTrader()
        stock_df = q_bot.train(stock_df, episodes=50)

        # backtest_strategy runs all 3 strategies and returns the hybrid results
        summary, trades, equity = backtest_strategy(stock_df)

        return jsonify({
            "summary": summary,
            "trades": trades.to_dict(orient="records"),
            "equity": equity.to_dict(orient="records")
        })

    except Exception as e:
        log.error(f"evaluation failed for {symbol}: {e}")
        return jsonify({"message": "Evaluation failed."}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5050)
