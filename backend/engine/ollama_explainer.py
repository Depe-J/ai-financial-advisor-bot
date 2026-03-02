import logging
log = logging.getLogger(__name__)

# ollama_explainer.py
# optional - uses a local Ollama LLM to generate richer explanations
# only works if you have Ollama running locally (ollama serve)
# set USE_OLLAMA = True in advisor.py to enable it

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"


def ollama_available() -> bool:
    try:
        r = requests.get("http://localhost:11434/", timeout=2)
        return r.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def llm_explanation(signal, short_ma, long_ma, symbol, confidence):
    # TODO: experiment with prompt wording - this version works but could be more natural
    prompt = (
        f"You are a financial advisor. In 2-3 sentences, explain why {symbol.upper()} has a {signal} signal. "
        f"Short-term MA: {short_ma:.2f}, long-term MA: {long_ma:.2f}, confidence: {confidence}%. "
        f"Keep it simple, no disclaimers."
    )

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=15
        )
        data = resp.json()
        return data.get("response", "").strip()
    except Exception as e:
        log.warning(f" Ollama unavailable: {e}")
        return ""
