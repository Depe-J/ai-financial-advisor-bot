# Design Decisions

Notes on the non-obvious choices made during development. Mainly for my own reference but also answers the "why did you do it that way" question.

---

## Why tabular Q-learning and not DQN?

I looked at DQN early on but the training was really unstable on 6 months of daily data. With only ~125 trading days the replay buffer never got big enough to stabilise. Tabular Q-learning converges in 10-20 episodes on this dataset and I can actually inspect the Q-table to see what the agent learned, which matters for the explainability side of the project.

The state space being only 3 values (Above/Below/Equal) makes tabular feasible. A continuous state representation would need a neural network but would also make the explanation engine much harder to build.

---

## Why 10-day and 50-day moving averages?

SMA10/SMA50 is a common short/mid-term crossover pair in retail trading. I tried SMA5/SMA20 first but it generated too many signals on a 6-month window (the agent was trading almost every day which made the transaction costs eat into returns). SMA10/SMA50 gave a reasonable trade frequency.

---

## Why does the advice endpoint use 6 months?

Practical constraint — yfinance returns clean daily OHLCV data reliably for 6mo without any gaps or adjusted-price weirdness. Longer windows introduced some data quality issues with certain tickers. It's also a realistic window for a short-term retail trading strategy. The advice endpoint only needs recent data to generate the current BUY/SELL/HOLD signal, so 6 months is the right call there.

## Why does the evaluation endpoint use 2 years?

The backtester needs a longer window to produce meaningful results. 2 years gives roughly 500 trading days. Of that, 70% (~350 days) is used to train the RL agent and the final 30% (~150 days) is held out as the test window — the backtest only runs on that held-out portion so there's no look-ahead bias. Running the full pipeline on 6 months would give too few trades to draw any real conclusions from. The confidence threshold calibration (0-20 low, 21-50 medium, >50 high) was also derived from 2 years of MA divergence data across AAPL, TSLA and NFLX.

---

## The hybrid fallback design

The Q-agent with 3 states converges quickly but tends to get stuck outputting HOLD for everything once it figures out that selling at the wrong time is punished heavily. The MA signal as a fallback for HOLD states stops this from degenerating. It also means the system degrades gracefully — if the RL agent is uncertain, it falls back to a rule that at least makes directional sense.

---

## Why not use a proper backtesting library like Backtrader?

Backtrader is great but it's a lot of configuration for what I needed. I wanted full control over the 70/30 split logic, the transaction cost model, and what gets returned to the frontend. Rolling my own meant I could return structured JSON directly without converting Backtrader's output format.

---

## Confidence score thresholds (0-20 low, 21-50 medium, >50 high)

These came from plotting the MA divergence distribution across AAPL, TSLA and NFLX over 2 years. The divergence rarely exceeds 5-8% in normal conditions. Anything above 50% only happens during strong trending markets so calling that "high confidence" felt right. Below 20% is basically noise.

---

## Why Ollama rather than OpenAI API?

Privacy — the app runs locally and I didn't want to send user stock queries to a third-party API. Ollama lets you run Mistral or Llama locally. The template engine is the default because it produces consistent results for evaluation; Ollama is opt-in.
