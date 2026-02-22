"""
NLP Sentiment Alpha from Financial Text
=========================================
Systematic alpha from text data (earnings calls, news, SEC filings).

The NLP alpha pipeline:
  1. Text ingestion: 8-K filings, earnings call transcripts, news
  2. Preprocessing: tokenise, lowercase, remove stopwords, stem/lemmatise
  3. Sentiment scoring: lexicon-based (fast) or ML-based (more accurate)
  4. Signal construction: cross-sectional rank → alpha signal
  5. Backtest: IC, Q5-Q1 spread

Approaches (in order of complexity):

1. Lexicon-based (Loughran-McDonald 2011):
   Count positive/negative words from a finance-specific dictionary.
   Net sentiment = (#positive - #negative) / #total_words
   Simple, interpretable, no model to overfit.
   LM is BETTER than Harvard GI for finance (GI miscategorises finance terms).

2. TF-IDF (Term Frequency – Inverse Document Frequency):
   Identifies words that are unusual vs. typical filings.
   Unusual language = uncertainty, hedging → negative signal.
   TF-IDF(t,d) = tf(t,d) · log(N/df(t))

3. Topic models (LDA):
   Discovers latent "topics" in corpus.
   Each document = mixture of topics.
   Topic weight change vs. expectation = signal.

4. BERT / FinBERT fine-tuning:
   Pre-trained transformer + fine-tuned on financial sentiment.
   State-of-the-art accuracy, but complex to deploy.

Key academic findings:
  - Tetlock (2007): negative WSJ media pessimism predicts lower stock returns
  - Loughran & McDonald (2011): MD&A negativity predicts future excess returns
  - Chen et al. (2014): Seeking Alpha articles predict returns
  - Huang et al. (2018): FinBERT outperforms LM on sentiment classification

Signal properties:
  - Earnings call sentiment: short-lived signal (5-10 day horizon)
  - 10-K negativity: slow decay (1-3 month horizon)
  - News sentiment: ultra-short (intraday to 1 day)
  - Tone change (vs previous filing): stronger signal than absolute tone

References:
  - Tetlock, P.C. (2007). Giving Content to Investor Sentiment. JF 62(3).
  - Loughran, T. & McDonald, B. (2011). When is a Liability Not a Liability?
    JF 66(1), 35–65.
  - Huang, A. et al. (2018). FinBERT: A Pre-trained Financial Language Model.
    arXiv:1908.10063.
  - Ke, Z. et al. (2019). Predicting Returns with Text Data. NBER WP 26186.
"""

import numpy as np
import pandas as pd
import re
from collections import Counter
from math import log
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Loughran-McDonald Finance Lexicon (condensed)
# ---------------------------------------------------------------------------

# A representative subset — full LM dictionary has ~2700 negative words
LM_NEGATIVE = {
    'loss', 'losses', 'decline', 'decreased', 'weak', 'weakened',
    'adverse', 'unfavorable', 'impairment', 'write-down', 'writedown',
    'restructuring', 'litigation', 'uncertainty', 'uncertainties',
    'risk', 'risks', 'failure', 'failed', 'default', 'bankruptcy',
    'negative', 'difficult', 'difficulty', 'challenging', 'challenges',
    'deterioration', 'deteriorated', 'reduced', 'reduction', 'shortfall',
    'concern', 'concerns', 'delay', 'delays', 'penalty', 'penalties',
    'volatile', 'volatility', 'headwind', 'headwinds', 'pressure',
    'slowing', 'slowdown', 'contraction', 'contracted', 'weaker',
}

LM_POSITIVE = {
    'growth', 'grew', 'increase', 'increased', 'strong', 'strengthened',
    'improvement', 'improved', 'benefit', 'beneficial', 'favorable',
    'opportunity', 'opportunities', 'exceeded', 'outperformed', 'robust',
    'resilient', 'recovery', 'recovered', 'record', 'positive', 'momentum',
    'expanding', 'expansion', 'efficient', 'efficiency', 'progress',
    'innovative', 'innovation', 'leading', 'leader', 'best', 'excellent',
    'accelerating', 'acceleration', 'successful', 'success', 'confident',
}

LM_UNCERTAINTY = {
    'approximately', 'uncertain', 'uncertainty', 'unclear', 'unknown',
    'possibly', 'might', 'could', 'may', 'perhaps', 'estimate', 'estimated',
    'believe', 'expects', 'anticipates', 'intends', 'forward-looking',
    'guidance', 'assumption', 'assume', 'projected', 'projection',
}


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

STOP_WORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall',
    'should', 'may', 'might', 'can', 'could', 'this', 'that', 'these',
    'those', 'it', 'its', 'we', 'our', 'us', 'they', 'them', 'their',
    'i', 'my', 'me', 'you', 'your', 'he', 'she', 'his', 'her',
}


def preprocess_text(text: str) -> list[str]:
    """
    Tokenise and preprocess financial text.
    Steps: lowercase → remove numbers/punctuation → tokenise → remove stopwords
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s\-]', ' ', text)
    tokens = text.split()
    tokens = [t.strip('-') for t in tokens if len(t) > 2 and t not in STOP_WORDS]
    return tokens


def loughran_mcdonald_score(tokens: list[str]) -> dict:
    """
    Compute Loughran-McDonald sentiment scores.
    Returns net sentiment, positive%, negative%, uncertainty%.
    """
    n = len(tokens) + 1e-10
    pos  = sum(1 for t in tokens if t in LM_POSITIVE)
    neg  = sum(1 for t in tokens if t in LM_NEGATIVE)
    unc  = sum(1 for t in tokens if t in LM_UNCERTAINTY)
    
    return {
        'net_sentiment':   (pos - neg) / n,
        'positive_pct':    pos / n,
        'negative_pct':    neg / n,
        'uncertainty_pct': unc / n,
        'n_words':         len(tokens),
        'n_positive':      pos,
        'n_negative':      neg,
    }


# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------

class TFIDF:
    """
    TF-IDF vectoriser for a corpus of financial documents.
    Useful for detecting unusual language vs. baseline filings.
    """
    
    def __init__(self, max_features: int = 500, min_df: int = 2):
        self.max_features = max_features
        self.min_df = min_df
        self.vocab = {}
        self.idf = {}
        self.n_docs = 0
    
    def fit(self, corpus: list[list[str]]) -> None:
        """Build vocabulary and compute IDF weights from corpus."""
        self.n_docs = len(corpus)
        
        # Document frequency
        df = Counter()
        for tokens in corpus:
            df.update(set(tokens))
        
        # Filter by min_df
        vocab_words = [w for w, c in df.most_common(self.max_features * 2)
                       if c >= self.min_df][:self.max_features]
        
        self.vocab = {w: i for i, w in enumerate(vocab_words)}
        self.idf = {w: log(self.n_docs / df[w]) for w in vocab_words}
    
    def transform(self, tokens: list[str]) -> dict:
        """Compute TF-IDF vector for a document."""
        n = len(tokens) + 1e-10
        tf = Counter(tokens)
        
        scores = {}
        for w, i in self.vocab.items():
            if w in tf:
                scores[w] = (tf[w] / n) * self.idf.get(w, 0)
        return scores
    
    def top_unusual_words(self, tokens: list[str], top_n: int = 10) -> list[tuple]:
        """Words with highest TF-IDF (most unusual vs. corpus)."""
        scores = self.transform(tokens)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]


# ---------------------------------------------------------------------------
# Earnings call signal construction
# ---------------------------------------------------------------------------

def construct_sentiment_signal(
    calls_df: pd.DataFrame,    # columns: date, ticker, text, fwd_return
    lookback_calls: int = 4,   # number of prior calls for baseline
) -> pd.DataFrame:
    """
    Build earnings call sentiment alpha signal.
    
    Signal construction:
      1. Score each call with LM sentiment
      2. Compute TONE CHANGE vs. prior 4 calls (change = stronger signal)
      3. Cross-sectional rank → z-score → alpha signal
    
    Tone change matters more than absolute tone because:
      - Analysts adapt to persistent corporate tone
      - Unexpected changes carry information
    """
    results = []
    
    for _, row in calls_df.iterrows():
        tokens = preprocess_text(row['text'])
        scores = loughran_mcdonald_score(tokens)
        
        # Historical baseline (prior calls for this ticker)
        ticker_hist = calls_df[
            (calls_df['ticker'] == row['ticker']) &
            (calls_df['date'] < row['date'])
        ].tail(lookback_calls)
        
        if len(ticker_hist) > 0:
            hist_scores = []
            for _, hist_row in ticker_hist.iterrows():
                hist_tok = preprocess_text(hist_row['text'])
                hist_scores.append(loughran_mcdonald_score(hist_tok)['net_sentiment'])
            baseline = np.mean(hist_scores)
        else:
            baseline = 0.0
        
        tone_change = scores['net_sentiment'] - baseline
        
        results.append({
            'date': row['date'],
            'ticker': row['ticker'],
            'net_sentiment': scores['net_sentiment'],
            'tone_change': tone_change,
            'uncertainty': scores['uncertainty_pct'],
            'fwd_return': row.get('fwd_return', np.nan),
        })
    
    df = pd.DataFrame(results)
    
    # Cross-sectional z-score (rank within each date)
    def zscore(x):
        s = x.std()
        return (x - x.mean()) / (s + 1e-10) if s > 0 else x * 0
    
    df['signal_zscore'] = df.groupby('date')['tone_change'].transform(zscore)
    return df


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

def generate_earnings_calls(n_stocks: int = 30, n_quarters: int = 16, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic earnings call transcripts with embedded sentiment.
    True signal: high positive tone change → higher 5-day return.
    """
    rng = np.random.default_rng(seed)
    
    positive_phrases = [
        "We delivered record revenue growth this quarter",
        "Our strong execution drove exceptional results",
        "We are seeing robust demand across all segments",
        "Margins improved significantly due to operational efficiency",
        "We exceeded our guidance and raised outlook",
        "Customer momentum continues to accelerate",
        "We are confident in our competitive position",
    ]
    
    negative_phrases = [
        "We faced significant headwinds in the quarter",
        "Revenue declined due to challenging market conditions",
        "Uncertainties in the macroeconomic environment remain",
        "We are taking restructuring charges this quarter",
        "Demand was weaker than anticipated across our segments",
        "We are revising our guidance lower for the year",
        "Supply chain pressures continue to impact margins",
    ]
    
    neutral_phrases = [
        "We continued to execute on our strategic priorities",
        "The quarter was in line with our expectations",
        "We made progress on our operational initiatives",
        "Our teams delivered on our commitments",
    ]
    
    rows = []
    dates = pd.date_range('2020-01-01', periods=n_quarters, freq='QE')
    tickers = [f'TICK{i:02d}' for i in range(n_stocks)]
    
    for date in dates:
        for ticker in tickers:
            # True sentiment: drawn from N(0, 1)
            true_sentiment = rng.normal(0, 1)
            
            # Build text with correlated sentiment
            phrases = []
            if true_sentiment > 0.5:
                n_pos = int(3 + true_sentiment * 2)
                n_neg = 1
            elif true_sentiment < -0.5:
                n_pos = 1
                n_neg = int(3 + abs(true_sentiment) * 2)
            else:
                n_pos, n_neg = 2, 2
            
            for _ in range(n_pos):
                phrases.append(rng.choice(positive_phrases))
            for _ in range(n_neg):
                phrases.append(rng.choice(negative_phrases))
            for _ in range(3):
                phrases.append(rng.choice(neutral_phrases))
            
            rng.shuffle(phrases)
            text = '. '.join(phrases)
            
            # Forward 5-day return with true signal + noise
            fwd_return = 0.02 * true_sentiment + rng.normal(0, 0.05)
            
            rows.append({
                'date': date,
                'ticker': ticker,
                'text': text,
                'true_sentiment': true_sentiment,
                'fwd_return': fwd_return,
            })
    
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("═" * 68)
    print("  NLP Sentiment Alpha from Earnings Calls")
    print("  Loughran-McDonald lexicon, TF-IDF, signal construction")
    print("═" * 68)
    
    # ── 1. Sentiment scoring demo ─────────────────────────────────
    print(f"\n── 1. Loughran-McDonald Sentiment Scoring ──")
    
    sample_texts = [
        ("Positive call",
         "We delivered record revenue growth. Strong execution drove exceptional results. "
         "Robust demand and improved margins exceeded guidance. Confident in our outlook."),
        ("Negative call",
         "We faced significant headwinds. Revenue declined due to challenging conditions. "
         "Uncertainty remains high. Restructuring charges and reduced guidance for the year."),
        ("Mixed/neutral",
         "We continued to execute on our strategic priorities. The quarter was in line with "
         "expectations. We made progress on our initiatives while managing risks carefully."),
    ]
    
    print(f"\n  {'Label':>16} {'Net Sent':>10} {'Pos%':>8} {'Neg%':>8} {'Unc%':>8} {'Words':>8}")
    print("  " + "─" * 56)
    for label, text in sample_texts:
        tokens = preprocess_text(text)
        scores = loughran_mcdonald_score(tokens)
        print(f"  {label:>16} {scores['net_sentiment']:>10.4f} "
              f"{scores['positive_pct']:>8.4f} {scores['negative_pct']:>8.4f} "
              f"{scores['uncertainty_pct']:>8.4f} {scores['n_words']:>8}")
    
    # ── 2. TF-IDF on corpus ───────────────────────────────────────
    print(f"\n── 2. TF-IDF — Unusual Language Detection ──")
    
    corpus_texts = [text for _, text in sample_texts]
    corpus_tokens = [preprocess_text(t) for t in corpus_texts]
    
    tfidf = TFIDF(max_features=50, min_df=1)
    tfidf.fit(corpus_tokens)
    
    for label, text in sample_texts[:2]:
        tokens = preprocess_text(text)
        top_words = tfidf.top_unusual_words(tokens, top_n=5)
        words_str = ', '.join([f"{w}({s:.3f})" for w, s in top_words])
        print(f"  {label}: {words_str}")
    
    # ── 3. Full alpha signal construction ─────────────────────────
    print(f"\n── 3. Earnings Call Alpha Signal (30 stocks × 16 quarters) ──")
    
    calls_df = generate_earnings_calls(n_stocks=30, n_quarters=16)
    
    print(f"  {len(calls_df)} earnings calls generated")
    
    signal_df = construct_sentiment_signal(calls_df)
    
    # Evaluate: IC of signal vs forward return
    valid = signal_df.dropna(subset=['signal_zscore', 'fwd_return'])
    
    from scipy.stats import spearmanr
    ic_net, _  = spearmanr(valid['net_sentiment'], valid['fwd_return'])
    ic_chg, _  = spearmanr(valid['tone_change'],   valid['fwd_return'])
    ic_sig, _  = spearmanr(valid['signal_zscore'], valid['fwd_return'])
    
    print(f"\n  {'Signal':>22} {'IC vs fwd_return':>18}")
    print("  " + "─" * 42)
    print(f"  {'Net sentiment':>22} {ic_net:>18.4f}")
    print(f"  {'Tone change':>22} {ic_chg:>18.4f}")
    print(f"  {'Cross-sect z-score':>22} {ic_sig:>18.4f}")
    
    # Quintile analysis
    print(f"\n── 4. Quintile Return Analysis ──")
    valid = valid.copy()
    valid['quintile'] = pd.qcut(valid['signal_zscore'], 5, labels=False, duplicates='drop')
    q_stats = valid.groupby('quintile')['fwd_return'].agg(['mean', 'std', 'count'])
    q_stats.index = q_stats.index + 1
    
    print(f"\n  {'Quintile':>10} {'Mean Ret':>12} {'Std':>10} {'Count':>8}")
    print("  " + "─" * 44)
    for q, row in q_stats.iterrows():
        bar = "█" * max(0, int(row['mean'] * 200 + 10))
        print(f"  {q:>10} {row['mean']:>12.4%} {row['std']:>10.4%} {row['count']:>8.0f} {bar}")
    
    spread = q_stats.loc[5, 'mean'] - q_stats.loc[1, 'mean']
    print(f"\n  Q5 - Q1 spread: {spread:.4%}")
    print(f"  (5-day return following earnings call)")
    
    print(f"""
── NLP Alpha in Production ──

  Data sources ranked by signal quality:
    1. Earnings call Q&A section (most informative — less scripted)
    2. MD&A section of 10-K/10-Q (annual/quarterly filings)
    3. Press releases and 8-K filings (immediate price reaction)
    4. News articles (fast but crowded — everyone has it)

  Key signal construction choices:
    - Tone CHANGE vs. prior quarter > absolute tone
    - Q&A tone > prepared remarks (harder to script)
    - Negative words are stronger predictor than positive (asymmetry)
    - Uncertainty words predict future volatility (not returns)

  FinBERT (state-of-the-art):
    from transformers import pipeline
    nlp = pipeline('sentiment-analysis', model='ProsusAI/finbert')
    result = nlp("We delivered record revenue growth")
    # → {{'label': 'positive', 'score': 0.97}}

  Interview question (Man AHL, Two Sigma):
  Q: "How would you build an alpha signal from earnings calls?"
  A: "1. Collect call transcripts (Refinitiv, Bloomberg, LSEG)
      2. Score with Loughran-McDonald — not Harvard GI (finance-specific)
      3. Compute tone CHANGE vs prior 4 calls (adapts to company style)
      4. Cross-sectional rank within each earnings date
      5. Use 5-10 day forward return as target
      6. Backtest with proper event-time alignment (no look-ahead)"
    """)
