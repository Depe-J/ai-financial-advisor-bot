import logging
import yfinance as yf
import pandas as pd

log = logging.getLogger(__name__)

def get_stock_data(ticker: str, period="6mo", interval="1d") -> pd.DataFrame:
    # using Ticker().history() rather than yf.download() - avoids the MultiIndex mess
    # auto_adjust=False keeps the raw prices, we dont want yfinance adjusting things silently
    try:
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(period=period, interval=interval, auto_adjust=False)

        if data.empty:
            raise ValueError("no data returned")

        # reset the date index so its just a regular column we can work with
        data.reset_index(inplace=True)
        data.columns = [col.lower() for col in data.columns]

        # make sure we have everything we need before passing it downstream
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_cols if col not in data.columns]
        if missing:
            raise KeyError(f"missing columns: {missing}")

        return data[required_cols]

    except:
        log.warning(f"failed to fetch {ticker}")
        return pd.DataFrame()
