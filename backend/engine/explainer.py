# explainer.py
# handles the confidence scoring and natural language explanation generation
# the explanation engine is intentionally rule-based for reproducibility


def calculate_confidence(short_ma, long_ma):
    # work out how far apart the two moving averages are as a percentage
    # the bigger the gap the more confident we are in the signal
    if short_ma == 0:
        return 0
    difference = abs(short_ma - long_ma)
    ratio = difference / short_ma
    return round(min(ratio * 100, 100))


def label_confidence(score):
    # band the confidence score into 3 levels
    # 0-20 = low, 21-50 = medium, anything above 50 = high
    # these thresholds came from trial and error looking at signal quality
    if score <= 20:
        return "low"
    elif score <= 50:
        return "medium"
    else:
        return "high"


def natural_language_explanation(signal: str, short_ma: float, long_ma: float, symbol: str) -> str:
    # builds a plain-english explanation for the advice
    # varies the wording a bit so it doesnt sound repetitive each time
    confidence = calculate_confidence(short_ma, long_ma)
    confidence_label = label_confidence(confidence)
    signal = signal.upper()
    symbol = symbol.upper()

    intro = f"{symbol} is currently showing a {signal} signal based on moving average crossover analysis.\n"

    details = (
        f"The short-term MA is {short_ma:.2f}, while the long-term MA is {long_ma:.2f}, "
        f"yielding a technical confidence score of approximately {confidence}% ({confidence_label} confidence).\n"
    )

    if signal == "BUY":
        recommendation = {
            "low": "The upward crossover appears weak. Consider watching the trend further before entering.",
            "medium": "This upward crossover suggests some bullish strength. It may be a decent time to buy with awareness of volatility.",
            "high": "A strong BUY signal. The momentum indicates a solid bullish trend forming."
        }[confidence_label]

    elif signal == "SELL":
        recommendation = {
            "low": "The downward signal lacks strength. Selling now may be premature unless other risks support it.",
            "medium": "The downward crossover is modest. Trimming exposure could be wise.",
            "high": "A strong SELL signal. The trend suggests a weakening market and selling may be optimal."
        }[confidence_label]

    else:  # HOLD
        recommendation = {
            "low": "We're not seeing enough momentum for a move. Hold tight and monitor the next developments closely.",
            "medium": "The signal remains unclear. Holding may be safest as momentum builds.",
            "high": "Despite strong movement, opposing trends offset each other. Holding is prudent while awaiting clarity."
        }[confidence_label]

    return intro + details + recommendation
