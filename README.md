# Financial Advisor Bot
CM3070 Final Project — Gurdeep Juneja

A stock advisory web app that combines Q-learning with SMA crossover signals to generate BUY/SELL/HOLD recommendations with plain-English explanations. Includes a custom backtesting engine comparing three strategy configurations.

---

## Recent Changes

- **Hybrid strategy fix** — the hybrid signal construction was corrected so the RL agent can influence trading decisions independently of the MA signal, rather than only applying when both strategies already agreed
- **Experiment logging** — every backtest run is now recorded to `evaluation/experiment_log.csv` with ticker, strategy, return, Sharpe ratio, and bootstrap confidence intervals for retrospective comparison
- **Automated tests** — added `backend/tests/test_backtester.py` with property-based tests verifying hybrid signal correctness

## Note for Marker

The submitted video was recorded prior to the hybrid strategy fix and therefore shows the evaluation panel with the original results where Hybrid matched MA-Only for TSLA. The current codebase contains the corrected hybrid logic — running the system live against TSLA will show different Hybrid results from MA-Only as described in the final report.

---

## How to Run

### Backend
```bash
cd backend
pip install flask flask-cors pandas numpy yfinance requests
python app.py
```
Runs on http://localhost:5050

### Frontend
```bash
cd frontend
npm install
npm run dev
```
Runs on http://localhost:5173

---

## Running Tests

```bash
cd backend
python -m pytest tests/ -v
```

The key test is `test_hybrid_differs_from_ma`, which verifies that the Hybrid strategy produces signals independent of the MA-Only baseline.

---

## Running Again? Clear the Cache First

If you've run the project before, clear the Python cache before restarting to make sure any code changes are picked up properly:

```bash
cd backend
find . -type d -name __pycache__ -exec rm -rf {} +
python app.py
```

Without doing this, Python may load old compiled files and your changes won't take effect.

---

## Usage

Type a stock ticker (e.g. AAPL, TSLA) or a natural language query (e.g. "what about Tesla", "tell me about Apple") into the chat box. The system fetches 6 months of data, trains the RL agent, and returns a recommendation with a confidence score and explanation.

Click "Run Analysis" to open the evaluation panel, which shows the equity curve, trade log, and a side-by-side comparison of MA-Only vs RL-Only vs Hybrid performance.

---

## Tech Stack

| Layer | Tech |
|---|---|
| Backend | Python 3, Flask |
| Data | yfinance, pandas, numpy |
| Strategy | Tabular Q-Learning, SMA Crossover |
| Evaluation | Custom backtester, bootstrap CI |
| Frontend | React, Tailwind CSS, Recharts |
| Optional LLM | Ollama (local) |

---

## Project Structure

```
backend/
  app.py                    - Flask endpoints
  data/fetch_stock_data.py  - yfinance wrapper
  data/ticker_lookup.py     - resolves natural language input to tickers
  strategies/
    basic_strategy.py       - SMA crossover
    q_learning_strategy.py  - Q-learning agent
  evaluation/
    backtester.py           - backtesting engine
    metrics.py              - Sharpe, drawdown, bootstrap CI
  engine/
    advisor.py              - generates advice string
    explainer.py            - NLG explanation engine
    ollama_explainer.py     - optional LLM integration
  tests/
    test_backtester.py      - automated tests for hybrid signal correctness
frontend/
  src/components/
    ChatBot.jsx             - chat UI
    EvaluationPanel.jsx     - results panel
evaluation/
  user_study_results.csv    - Likert scores from user study
  analyse_user_study.py     - reproduces Table 4 from report
  experiment_log.csv        - auto-generated log of every backtest run (ticker, strategy, return, Sharpe, CI)
```

---

## Optional: Ollama LLM

Set `USE_OLLAMA = True` in `backend/engine/advisor.py` and run `ollama serve` locally. Falls back to template engine automatically if Ollama isn't running.
