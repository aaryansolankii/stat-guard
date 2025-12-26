DEFAULT_POLICY = {
    "min_sample_size": 30,
    "max_imbalance_ratio": 2.0,
    "max_missing_pct": 0.05,
    "max_skewness": 2.0,
}

STRICT_POLICY = {
    "min_sample_size": 50,
    "max_imbalance_ratio": 1.5,
    "max_missing_pct": 0.02,
    "max_skewness": 1.5,
}

POLICIES = {
    "default": DEFAULT_POLICY,
    "strict": STRICT_POLICY,
}
