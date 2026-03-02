import pandas as pd
# applies SMA crossover logic and generates BUY/SELL/HOLD signals for each day
def apply_moving_average_strategy(df: pd.DataFrame, short_window=10, long_window=50) -> pd.DataFrame:
    if df.empty or 'close' not in df.columns:
        print("no close column found, returning early")
        return df

    df = df.copy()

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

    # min_periods=1 so we get values from day 1 even before the full window exists
    df['sma_short'] = df['close'].rolling(window=short_window, min_periods=1).mean().round(2)
    df['sma_long']  = df['close'].rolling(window=long_window, min_periods=1).mean().round(2)

    df['signal'] = df.apply(
        lambda row: 1 if row['sma_short'] > row['sma_long'] else (-1 if row['sma_short'] < row['sma_long'] else 0),
        axis=1
    )

    # keep a copy before RL overwrites it - needed for the 3-way comparison
    df['ma_signal'] = df['signal']

    # signal strength = how far apart the two MAs are
    df['signal_strength'] = (df['sma_short'] - df['sma_long']).abs().round(2)

    return df
