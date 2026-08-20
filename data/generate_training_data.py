"""
Generate 1500 guideline-driven synthetic churn records for Retainr training.

Churn labels are derived from a continuous, feature-driven risk score for
every row (not a flat per-segment coin flip), so the resulting dataset is
learnable enough for a classifier to reach high accuracy while still
containing a small, intentional minority of genuinely ambiguous/noisy cases.

Eleven features: six engagement metrics plus five derived from interaction
history. The interaction features follow the same empty-window convention the
Apex ChurnFeatureBuilder uses, so training rows and live predictions describe
a silent account identically. See _apply_empty_window_convention.
"""

import os
import numpy as np
import pandas as pd

RANDOM_STATE = 42
N_RECORDS = 1500
AMBIGUOUS_FRAC = 0.04
SIGMOID_K = 13.0
SIGMOID_MIDPOINT = 0.42

# Recency is capped here exactly as it is in Apex, so 90 is both "three months
# quiet" and "never contacted at all".
MAX_DAYS_SINCE = 90

COLUMNS = [
    'login_frequency',
    'feature_usage_score',
    'support_ticket_count',
    'nps_score',
    'contract_value',
    'duration_months',
    'total_interactions_30d',
    'negative_sentiment_ratio',
    'avg_interaction_duration',
    'days_since_last_interaction',
    'support_interaction_ratio',
    'churned',
]

# Features that carry one decimal place rather than being whole numbers.
DECIMAL_COLUMNS = [
    'negative_sentiment_ratio',
    'avg_interaction_duration',
    'support_interaction_ratio',
]

SEGMENTS = {
    'Healthy': {
        'count': 450,
        'ranges': {
            'login_frequency': (35, 60),
            'feature_usage_score': (65, 95),
            'support_ticket_count': (0, 2),
            'nps_score': (7, 10),
            'contract_value': (40000, 90000),
            'duration_months': (18, 48),
            'total_interactions_30d': (15, 32),
            'negative_sentiment_ratio': (0, 15),
            'avg_interaction_duration': (25, 70),
            'days_since_last_interaction': (0, 7),
            'support_interaction_ratio': (0, 20),
        },
    },
    'At-Risk': {
        'count': 450,
        'ranges': {
            'login_frequency': (1, 14),
            'feature_usage_score': (5, 34),
            'support_ticket_count': (5, 15),
            'nps_score': (1, 3),
            'contract_value': (5000, 25000),
            'duration_months': (2, 12),
            # The 0 floor is what produces the empty-window rows.
            'total_interactions_30d': (0, 6),
            'negative_sentiment_ratio': (45, 100),
            'avg_interaction_duration': (3, 15),
            'days_since_last_interaction': (30, 90),
            'support_interaction_ratio': (50, 100),
        },
    },
    'Medium Risk': {
        'count': 375,
        'ranges': {
            'login_frequency': (15, 34),
            'feature_usage_score': (35, 64),
            'support_ticket_count': (2, 6),
            'nps_score': (4, 6),
            'contract_value': (20000, 50000),
            'duration_months': (8, 24),
            'total_interactions_30d': (6, 15),
            'negative_sentiment_ratio': (15, 45),
            'avg_interaction_duration': (12, 35),
            'days_since_last_interaction': (7, 25),
            'support_interaction_ratio': (20, 50),
        },
    },
    'Deceptive': {
        'count': 150,
        'ranges': {
            'login_frequency': (20, 39),
            'feature_usage_score': (45, 69),
            'support_ticket_count': (4, 8),
            'nps_score': (3, 5),
            'contract_value': (30000, 60000),
            'duration_months': (12, 30),
            # Volume and recency still look respectable. What gives these away is
            # that the contact is soured and mostly about support.
            'total_interactions_30d': (8, 18),
            'negative_sentiment_ratio': (35, 70),
            'avg_interaction_duration': (10, 25),
            'days_since_last_interaction': (5, 20),
            'support_interaction_ratio': (45, 80),
        },
    },
    'New Accounts': {
        'count': 75,
        'ranges': {
            'login_frequency': (10, 44),
            'feature_usage_score': (20, 59),
            'support_ticket_count': (1, 5),
            'nps_score': (4, 7),
            'contract_value': (15000, 45000),
            'duration_months': (1, 6),
            'total_interactions_30d': (3, 20),
            'negative_sentiment_ratio': (5, 35),
            'avg_interaction_duration': (15, 45),
            'days_since_last_interaction': (1, 20),
            'support_interaction_ratio': (10, 45),
        },
    },
}


def _randint(rng, low, high):
    """Inclusive integer draw."""
    return int(rng.integers(low, high + 1))


def _round_contract(value, low, high):
    rounded = int(round(value / 1000.0) * 1000)
    return int(np.clip(rounded, low, high))


def _uniform1(rng, low, high):
    """Draw to one decimal place, matching the scale of the Salesforce fields."""
    return round(float(rng.uniform(low, high)), 1)


def _sample_row(rng, ranges):
    login = _randint(rng, *ranges['login_frequency'])
    usage = _randint(rng, *ranges['feature_usage_score'])
    tickets = _randint(rng, *ranges['support_ticket_count'])
    nps = _randint(rng, *ranges['nps_score'])
    contract = _round_contract(
        _randint(rng, *ranges['contract_value']),
        ranges['contract_value'][0],
        ranges['contract_value'][1],
    )
    duration = _randint(rng, *ranges['duration_months'])
    return {
        'login_frequency': login,
        'feature_usage_score': usage,
        'support_ticket_count': tickets,
        'nps_score': nps,
        'contract_value': contract,
        'duration_months': duration,
        'total_interactions_30d': _randint(rng, *ranges['total_interactions_30d']),
        'negative_sentiment_ratio': _uniform1(rng, *ranges['negative_sentiment_ratio']),
        'avg_interaction_duration': _uniform1(rng, *ranges['avg_interaction_duration']),
        'days_since_last_interaction': _randint(rng, *ranges['days_since_last_interaction']),
        'support_interaction_ratio': _uniform1(rng, *ranges['support_interaction_ratio']),
    }


def _apply_empty_window_convention(row):
    """An account with nothing in the 30-day window has no ratios to report.

    Zeroing them is not the same as saying the account is happy, which is why
    recency is pinned to the cap: that is the only feature left carrying signal
    for a silent account. ChurnFeatureBuilder.build encodes the identical rule,
    and the two must not drift or a silent account will score differently live
    than it did in training.
    """
    if row['total_interactions_30d'] == 0:
        row['negative_sentiment_ratio'] = 0.0
        row['avg_interaction_duration'] = 0.0
        row['support_interaction_ratio'] = 0.0
        row['days_since_last_interaction'] = MAX_DAYS_SINCE
    return row


def _apply_correlations(row, segment_name, ranges, rng):
    """Nudge correlated features, then clip to segment bounds."""
    login = row['login_frequency']
    tickets = row['support_ticket_count']
    lo_u, hi_u = ranges['feature_usage_score']
    lo_n, hi_n = ranges['nps_score']

    if segment_name == 'Healthy':
        usage = login + _randint(rng, 20, 40)
        row['feature_usage_score'] = int(np.clip(usage, lo_u, hi_u))
    elif segment_name == 'At-Risk':
        nps = 5 - (tickets // 3) + _randint(rng, -1, 1)
        row['nps_score'] = int(np.clip(nps, lo_n, hi_n))
        usage = login + _randint(rng, 5, 20)
        row['feature_usage_score'] = int(np.clip(usage, lo_u, hi_u))
    else:
        usage = login + _randint(rng, 5, 25)
        row['feature_usage_score'] = int(np.clip(usage, lo_u, hi_u))
        nps = hi_n - (tickets // 3) + _randint(rng, -1, 1)
        row['nps_score'] = int(np.clip(nps, lo_n, hi_n))

    # Recency tracks volume. Accounts you talk to often were talked to recently,
    # so drawing the two independently would produce incoherent rows such as 30
    # interactions in the window but a last contact 80 days ago.
    lo_t, hi_t = ranges['total_interactions_30d']
    lo_d, hi_d = ranges['days_since_last_interaction']
    volume_fraction = (row['total_interactions_30d'] - lo_t) / max(hi_t - lo_t, 1)
    recency = hi_d - volume_fraction * (hi_d - lo_d) + _randint(rng, -3, 3)
    row['days_since_last_interaction'] = int(np.clip(round(recency), lo_d, hi_d))

    # Negative sentiment and support share rise together: the more of the
    # relationship that is tickets, the worse the tone of it tends to be.
    lo_s, hi_s = ranges['support_interaction_ratio']
    support = row['negative_sentiment_ratio'] * 0.8 + _randint(rng, -10, 10)
    row['support_interaction_ratio'] = round(float(np.clip(support, lo_s, hi_s)), 1)

    return _apply_empty_window_convention(row)


def _bernoulli(rng, p):
    return int(rng.random() < p)


def _risk_score(row):
    """Continuous, feature-driven churn propensity in [0, 1].

    Weights sum to 1.0. The two interaction features that overlap existing ones
    are deliberately small: support_interaction_ratio covers similar ground to
    support_ticket_count, and total_interactions_30d to login_frequency, so
    giving them real weight would just double-count the same behaviour.
    """
    login_risk = 1 - row['login_frequency'] / 60
    usage_risk = 1 - row['feature_usage_score'] / 100
    ticket_risk = row['support_ticket_count'] / 15
    nps_risk = 1 - (row['nps_score'] - 1) / 9
    contract_risk = 1 - (row['contract_value'] - 5000) / 95000
    duration_risk = 1 - row['duration_months'] / 48

    recency_risk = row['days_since_last_interaction'] / MAX_DAYS_SINCE
    sentiment_risk = row['negative_sentiment_ratio'] / 100
    support_share_risk = row['support_interaction_ratio'] / 100
    volume_risk = 1 - min(row['total_interactions_30d'], 30) / 30
    # Short conversations suggest a shallow relationship. Anything past an hour
    # is treated as equally deep rather than better still.
    depth_risk = 1 - min(row['avg_interaction_duration'], 60) / 60

    return (
        0.18 * nps_risk
        + 0.14 * recency_risk
        + 0.13 * ticket_risk
        + 0.13 * login_risk
        + 0.12 * sentiment_risk
        + 0.10 * usage_risk
        + 0.06 * contract_risk
        + 0.06 * duration_risk
        + 0.04 * support_share_risk
        + 0.02 * volume_risk
        + 0.02 * depth_risk
    )


def _score_to_prob(risk_score, k=SIGMOID_K, midpoint=SIGMOID_MIDPOINT):
    """Sharpened, left-shifted logistic: churn is more deterministic than
    retention (bad metrics almost always mean churn; good metrics don't
    guarantee retention)."""
    return 1.0 / (1.0 + np.exp(-k * (risk_score - midpoint)))


def _apply_cross_feature_rules(row, base_prob, rng):
    """Override probability using high-signal churn/retention rules.
    These represent combinations the guidelines describe as "strongly
    predictive", so they are treated as near-deterministic."""
    login = row['login_frequency']
    usage = row['feature_usage_score']
    tickets = row['support_ticket_count']
    nps = row['nps_score']
    contract = row['contract_value']
    duration = row['duration_months']

    total_interactions = row['total_interactions_30d']
    days_since = row['days_since_last_interaction']
    neg_sentiment = row['negative_sentiment_ratio']
    support_share = row['support_interaction_ratio']

    churn_hits = []
    if login < 10 and usage < 30:
        churn_hits.append(0.95)
    if nps <= 3 and tickets > 6:
        churn_hits.append(0.93)
    if duration < 6 and usage < 40:
        churn_hits.append(0.92)
    if login < 15 and nps <= 4 and tickets > 5:
        churn_hits.append(0.96)
    if contract < 15000 and duration < 8:
        churn_hits.append(0.92)
    # Gone quiet and unhappy about it when they last spoke.
    if days_since > 60 and neg_sentiment > 40:
        churn_hits.append(0.96)
    # The relationship has narrowed to firefighting.
    if support_share > 70 and neg_sentiment > 50:
        churn_hits.append(0.94)
    # Empty window. The ratios above are all zeroed by convention, so recency is
    # the only thing left to fire on, and it needs its own rule to register.
    if total_interactions == 0 and login < 15:
        churn_hits.append(0.93)

    retain_hits = []
    if login > 35 and usage > 70:
        retain_hits.append(0.97)
    if nps >= 8 and tickets <= 2:
        retain_hits.append(0.97)
    if duration > 24 and contract > 50000:
        retain_hits.append(0.96)
    if login > 40 and nps >= 7 and tickets <= 1:
        retain_hits.append(0.99)
    if usage > 80 and duration > 18:
        retain_hits.append(0.97)
    if days_since <= 7 and neg_sentiment < 10:
        retain_hits.append(0.96)
    if total_interactions > 20 and support_share < 20:
        retain_hits.append(0.95)

    prob = base_prob
    if churn_hits:
        prob = max(prob, max(churn_hits))
    if retain_hits:
        prob = min(prob, 1.0 - max(retain_hits))

    return prob


def _make_ambiguous_row(rng, rule_id):
    """Build an ambiguous Rule 11-14 style row whose probability is still
    derived from its own (mid-range) feature values, clipped to the rule's
    ambiguous band, rather than a flat feature-blind draw."""
    if rule_id == 11:
        row = {
            'login_frequency': _randint(rng, 21, 35),
            'feature_usage_score': _randint(rng, 41, 60),
            'support_ticket_count': _randint(rng, 3, 5),
            'nps_score': _randint(rng, 4, 6),
            'contract_value': _round_contract(_randint(rng, 20001, 50000), 5000, 100000),
            'duration_months': _randint(rng, 13, 24),
            'total_interactions_30d': _randint(rng, 7, 14),
            'negative_sentiment_ratio': _uniform1(rng, 20, 40),
            'avg_interaction_duration': _uniform1(rng, 15, 30),
            'days_since_last_interaction': _randint(rng, 10, 25),
            'support_interaction_ratio': _uniform1(rng, 25, 50),
        }
        band = (0.40, 0.50)
    elif rule_id == 12:
        row = {
            'login_frequency': _randint(rng, 36, 50),
            'feature_usage_score': _randint(rng, 61, 80),
            'support_ticket_count': _randint(rng, 2, 5),
            'nps_score': _randint(rng, 4, 5),
            'contract_value': _round_contract(_randint(rng, 30000, 70000), 5000, 100000),
            'duration_months': _randint(rng, 12, 30),
            'total_interactions_30d': _randint(rng, 10, 20),
            'negative_sentiment_ratio': _uniform1(rng, 30, 50),
            'avg_interaction_duration': _uniform1(rng, 18, 35),
            'days_since_last_interaction': _randint(rng, 5, 18),
            'support_interaction_ratio': _uniform1(rng, 30, 55),
        }
        band = (0.55, 0.65)
    elif rule_id == 13:
        row = {
            'login_frequency': _randint(rng, 36, 55),
            'feature_usage_score': _randint(rng, 61, 85),
            'support_ticket_count': _randint(rng, 0, 3),
            'nps_score': _randint(rng, 6, 9),
            'contract_value': _round_contract(_randint(rng, 20000, 60000), 5000, 100000),
            'duration_months': _randint(rng, 1, 6),
            'total_interactions_30d': _randint(rng, 12, 25),
            'negative_sentiment_ratio': _uniform1(rng, 5, 25),
            'avg_interaction_duration': _uniform1(rng, 20, 45),
            'days_since_last_interaction': _randint(rng, 1, 12),
            'support_interaction_ratio': _uniform1(rng, 10, 35),
        }
        band = (0.30, 0.40)
    else:
        row = {
            'login_frequency': _randint(rng, 25, 45),
            'feature_usage_score': _randint(rng, 50, 75),
            'support_ticket_count': _randint(rng, 6, 10),
            'nps_score': _randint(rng, 4, 7),
            'contract_value': _round_contract(_randint(rng, 50001, 100000), 5000, 100000),
            'duration_months': _randint(rng, 12, 36),
            'total_interactions_30d': _randint(rng, 8, 18),
            'negative_sentiment_ratio': _uniform1(rng, 25, 45),
            'avg_interaction_duration': _uniform1(rng, 12, 28),
            'days_since_last_interaction': _randint(rng, 8, 22),
            'support_interaction_ratio': _uniform1(rng, 40, 70),
        }
        band = (0.40, 0.50)

    row = _apply_empty_window_convention(row)
    risk_score = _risk_score(row)
    prob = _score_to_prob(risk_score)
    prob = float(np.clip(prob, band[0], band[1]))
    row['churned'] = _bernoulli(rng, prob)
    return row


def generate_dataset(n_records=N_RECORDS, random_state=RANDOM_STATE):
    rng = np.random.default_rng(random_state)
    assert sum(s['count'] for s in SEGMENTS.values()) == n_records

    rows = []
    segment_labels = []

    for name, cfg in SEGMENTS.items():
        ranges = cfg['ranges']
        for _ in range(cfg['count']):
            row = _sample_row(rng, ranges)
            row = _apply_correlations(row, name, ranges, rng)

            risk_score = _risk_score(row)
            base_prob = _score_to_prob(risk_score)
            prob = _apply_cross_feature_rules(row, base_prob, rng)

            row['churned'] = _bernoulli(rng, prob)
            rows.append(row)
            segment_labels.append(name)

    df = pd.DataFrame(rows)
    df['_segment'] = segment_labels

    # Budget-cuts noise: flip ~1.5% of Medium Risk retained accounts to churn
    medium_idx = df.index[df['_segment'] == 'Medium Risk'].tolist()
    flip_n = max(1, int(round(len(medium_idx) * 0.015)))
    flip_candidates = df.loc[medium_idx]
    retained = flip_candidates.index[flip_candidates['churned'] == 0].tolist()
    if retained:
        chosen = list(rng.choice(retained, size=min(flip_n, len(retained)), replace=False))
        df.loc[chosen, 'churned'] = 1

    # Inject ~6% ambiguous cases (overwrite random subset)
    n_ambiguous = int(round(n_records * AMBIGUOUS_FRAC))
    amb_idx = list(rng.choice(df.index, size=n_ambiguous, replace=False))
    for i, idx in enumerate(amb_idx):
        rule_id = 11 + (i % 4)
        amb = _make_ambiguous_row(rng, rule_id)
        for col in COLUMNS:
            df.at[idx, col] = amb[col]
        df.at[idx, '_segment'] = f'Ambiguous-{rule_id}'

    segment_summary = df['_segment'].value_counts()
    df = df[COLUMNS].copy()

    validate_dataframe(df)
    return df, segment_summary


def validate_dataframe(df):
    """Enforce data-quality rules; raise on violation."""
    assert len(df) == N_RECORDS, f"Expected {N_RECORDS} rows, got {len(df)}"
    assert df.isnull().sum().sum() == 0, "Null values found"
    assert set(df.columns) == set(COLUMNS)

    assert df['login_frequency'].between(0, 60).all()
    assert df['feature_usage_score'].between(0, 100).all()
    assert df['support_ticket_count'].between(0, 15).all()
    assert df['nps_score'].between(1, 10).all()
    assert df['contract_value'].between(5000, 100000).all()
    assert df['duration_months'].between(1, 48).all()
    assert df['churned'].isin([0, 1]).all()
    assert (df['contract_value'] % 1000 == 0).all(), "Contract values must be $1k increments"

    assert df['total_interactions_30d'].between(0, 40).all()
    assert df['negative_sentiment_ratio'].between(0, 100).all()
    assert df['avg_interaction_duration'].between(0, 120).all()
    assert df['days_since_last_interaction'].between(0, MAX_DAYS_SINCE).all()
    assert df['support_interaction_ratio'].between(0, 100).all()

    # The empty-window convention has to hold for every row, or a silent account
    # scored live will not resemble anything the model was trained on.
    empty = df['total_interactions_30d'] == 0
    assert (df.loc[empty, 'days_since_last_interaction'] == MAX_DAYS_SINCE).all(), \
        "Rows with no interactions must report the recency cap"
    for column in ['negative_sentiment_ratio', 'avg_interaction_duration',
                   'support_interaction_ratio']:
        assert (df.loc[empty, column] == 0).all(), \
            f"Rows with no interactions must report 0 for {column}"


def main():
    print("=" * 50)
    print("RETAINR SYNTHETIC TRAINING DATA GENERATOR")
    print("=" * 50)

    df, segment_summary = generate_dataset()

    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_data.csv')
    df.to_csv(out_path, index=False)

    churn_rate = df['churned'].mean()
    empty_window = int((df['total_interactions_30d'] == 0).sum())
    print(f"\nWrote {len(df)} records to {out_path}")
    print(f"\nSegment / source mix:\n{segment_summary}")
    print(f"\nChurn distribution:\n{df['churned'].value_counts()}")
    print(f"Churn rate: {churn_rate:.1%}")
    print(f"Empty-interaction-window rows: {empty_window} ({empty_window / len(df):.1%})")
    print("\nGeneration complete!")


if __name__ == '__main__':
    main()
