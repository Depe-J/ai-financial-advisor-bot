# analyse_user_study.py
# reads the Likert scores from the user study and prints the aggregated table
# run with: python evaluation/analyse_user_study.py

import pandas as pd
import os

CSV_PATH = os.path.join(os.path.dirname(__file__), "user_study_results.csv")


def main():
    df = pd.read_csv(CSV_PATH)

    print("=== Aggregated Likert Results ===\n")

    modes = {
        "Numeric Only":      ("numeric_clarity",  "numeric_trust",  "numeric_confidence"),
        "Template XAI":      ("template_clarity", "template_trust", "template_confidence"),
        "LLM-Enhanced XAI":  ("llm_clarity",      "llm_trust",      "llm_confidence"),
    }

    rows = []
    for mode, (c, t, conf) in modes.items():
        rows.append({
            "Mode":           mode,
            "Clarity / 5":    round(df[c].mean(), 1),
            "Trust / 5":      round(df[t].mean(), 1),
            "Confidence / 5": round(df[conf].mean(), 1),
        })

    results = pd.DataFrame(rows)
    print(results.to_string(index=False))

    print("\n=== Preferred Mode Distribution ===\n")
    print(df['preferred_mode'].value_counts().to_string())

    llm_pref = (df['preferred_mode'] == 'LLM-Enhanced').sum()
    pct = round(llm_pref / len(df) * 100)
    print(f"\n{pct}% of participants preferred LLM-Enhanced explanations")


if __name__ == "__main__":
    main()
