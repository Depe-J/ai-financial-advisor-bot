import logging
import pandas as pd
from engine.explainer import natural_language_explanation, calculate_confidence, label_confidence
from engine.ollama_explainer import ollama_available, llm_explanation

log = logging.getLogger(__name__)

# set to True to use the local Ollama LLM for richer explanations
# falls back to template engine automatically if Ollama isn't running
USE_OLLAMA = False


def generate_advice(df: pd.DataFrame, symbol: str = "AAPL") -> str:
    # takes the final signal from the strategy and turns it into readable advice

    if df.empty or 'signal' not in df.columns:
        log.warning(f"generate_advice called with no signal data for {symbol}")
        return f"Sorry, couldn't generate a recommendation for {symbol} right now."

    # get the most recent row - thats the current recommendation
    latest = df.iloc[-1]
    signal = int(latest['signal'])

    # map the numeric signal back to a human readable action
    # 1 = buy, -1 = sell, anything else defaults to hold
    if signal == 1:
        action = "BUY"
    elif signal == -1:
        action = "SELL"
    else:
        action = "HOLD"

    short_ma = float(latest['sma_short'])
    long_ma = float(latest['sma_long'])

    # try LLM first if enabled and Ollama is actually reachable
    explanation = ""
    if USE_OLLAMA and ollama_available():
        confidence = calculate_confidence(short_ma, long_ma)
        explanation = llm_explanation(action, short_ma, long_ma, symbol, confidence)

    # fall back to deterministic template engine if LLM failed or isnt enabled
    # this is the default path for most users
    if not explanation:
        explanation = natural_language_explanation(
            signal=action,
            short_ma=short_ma,
            long_ma=long_ma,
            symbol=symbol
        )

    return f"Recommendation: {action} {symbol.upper()}\n\n{explanation}"
