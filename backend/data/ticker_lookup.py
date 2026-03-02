import os
import pandas as pd

# loads the ticker list once at startup and keeps it in memory
_df = None

def _load():
    global _df
    if _df is None:
        csv_path = os.path.join(os.path.dirname(__file__), 'tickers.csv')
        _df = pd.read_csv(csv_path)
        _df['ticker'] = _df['ticker'].str.strip().str.upper()
        _df['company'] = _df['company'].str.strip()
        # add a lowercase version of company name for matching
        _df['company_lower'] = _df['company'].str.lower()
    return _df


def resolve_ticker(raw_input: str) -> str | None:
    """
    Takes raw user input and returns the best matching ticker symbol.

    Matching priority:
    1. Exact ticker match (e.g. "AAPL" -> "AAPL")
    2. Exact company name match (e.g. "Apple" -> "AAPL")
    3. Company name starts with input (e.g. "Tesla" -> "TSLA")
    4. Company name contains input (e.g. "microsoft" -> "MSFT")
    5. Individual word matching with stopwords filter

    Returns None if no match found.
    """
    df = _load()
    query = raw_input.strip()
    query_upper = query.upper()
    query_lower = query.lower()

    # 1. exact ticker match
    exact = df[df['ticker'] == query_upper]
    if not exact.empty:
        return exact.iloc[0]['ticker']

    # 2. exact company name match (case-insensitive)
    name_exact = df[df['company_lower'] == query_lower]
    if not name_exact.empty:
        return name_exact.iloc[0]['ticker']

    # 3. company name starts with input
    name_starts = df[df['company_lower'].str.startswith(query_lower)]
    if not name_starts.empty:
        return name_starts.iloc[0]['ticker']

    # 4. company name contains input
    name_contains = df[df['company_lower'].str.contains(query_lower, regex=False)]
    if not name_contains.empty:
        return name_contains.iloc[0]['ticker']

    # 5. try matching individual words from multi-word input
    # common English words that are NOT useful as ticker lookups
    stopwords = {
        # question/conversation words
        'what', 'when', 'where', 'who', 'why', 'how', 'which', 'that', 'this',
        'these', 'those', 'there', 'here',
        # verbs
        'tell', 'give', 'show', 'get', 'let', 'think', 'know', 'want', 'need',
        'does', 'did', 'has', 'have', 'had', 'was', 'were', 'are', 'will',
        'would', 'could', 'should', 'can', 'may', 'might', 'invest', 'investing',
        'buy', 'sell', 'hold', 'trade', 'trading',
        # pronouns
        'the', 'a', 'an', 'i', 'you', 'we', 'he', 'she', 'it', 'they', 'me',
        'him', 'her', 'its', 'our', 'your', 'their', 'my',
        # prepositions/conjunctions
        'about', 'from', 'into', 'onto', 'with', 'without', 'for', 'and', 'but',
        'or', 'nor', 'so', 'yet', 'both', 'either', 'not', 'also', 'just',
        'then', 'than', 'too', 'very', 'some', 'more', 'most', 'such',
        # common nouns that aren't company names
        'stock', 'stocks', 'share', 'shares', 'company', 'companies', 'market',
        'please', 'thanks', 'okay', 'yes', 'no',
        # financial context words that cause false matches (e.g. "fund" -> OFLX)
        'fund', 'funds', 'retirement', 'pension', 'portfolio', 'money', 'cash',
        'price', 'returns', 'dividend', 'growth', 'value', 'risk',
        # celebrity names often used with tickers
        'elon', 'musk', 'musks',
    }

    for word in query.split():
        word_clean = word.strip("'s,.")
        # increased minimum length from 2 to 3 to avoid short ambiguous words causing false matches
        # e.g. "APPL" typo was matching against AAOI instead of AAPL
        if len(word_clean) > 3 and word_clean.lower() not in stopwords:
            # check if the word is itself a valid ticker before trying company name matching
            # this handles inputs like "tell me about MSFT" where MSFT is a ticker not a company name
            ticker_match = df[df['ticker'] == word_clean.upper()]
            if not ticker_match.empty:
                return ticker_match.iloc[0]['ticker']
            # otherwise try matching against company names as before
            word_lower = word_clean.lower()
            match = df[df['company_lower'].str.startswith(word_lower)]
            if not match.empty:
                return match.iloc[0]['ticker']

    return None
