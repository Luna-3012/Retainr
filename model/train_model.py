"""
Retainr Churn Prediction Model
Trains a Random Forest classifier on customer engagement data and saves the model for API serving.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
)
import joblib
import os

FEATURE_COLUMNS = [
    'login_frequency',
    'feature_usage_score',
    'support_ticket_count',
    'nps_score',
    'contract_value',
    'duration_months',
]


def load_data(filepath=None):
    """Load training data from CSV."""
    if filepath is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        filepath = os.path.join(project_root, 'data', 'training_data.csv')

    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} records")
    print(f"Churn distribution:\n{df['churned'].value_counts()}")
    return df


def validate_dataframe(df):
    """Assert required columns, no nulls, and guideline ranges."""
    required = FEATURE_COLUMNS + ['churned']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df[required].isnull().any().any():
        raise ValueError("Training data contains null values")

    checks = [
        (df['login_frequency'].between(0, 60).all(), 'login_frequency must be 0–60'),
        (df['feature_usage_score'].between(0, 100).all(), 'feature_usage_score must be 0–100'),
        (df['support_ticket_count'].between(0, 15).all(), 'support_ticket_count must be 0–15'),
        (df['nps_score'].between(1, 10).all(), 'nps_score must be 1–10'),
        (df['contract_value'].between(5000, 100000).all(), 'contract_value must be 5000–100000'),
        (df['duration_months'].between(1, 48).all(), 'duration_months must be 1–48'),
        (df['churned'].isin([0, 1]).all(), 'churned must be 0 or 1'),
    ]
    for ok, msg in checks:
        if not ok:
            raise ValueError(msg)


def prepare_features(df):
    """Prepare feature matrix and target variable."""
    X = df[FEATURE_COLUMNS]
    y = df['churned']
    return X, y, FEATURE_COLUMNS


def print_proba_histogram(y_proba, bins=10):
    """Print-only histogram of predicted churn probabilities."""
    counts, edges = np.histogram(y_proba, bins=bins, range=(0.0, 1.0))
    print("\nPrediction probability histogram (test set):")
    max_count = max(counts) if len(counts) else 1
    for i, count in enumerate(counts):
        lo, hi = edges[i], edges[i + 1]
        bar = '#' * int(30 * count / max_count) if max_count else ''
        print(f"  [{lo:.1f}, {hi:.1f}): {count:3d} {bar}")


def train_model(X, y):
    """Train Random Forest model (no scaling — trees use raw features)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced',
        oob_score=True,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.2f}")
    print(f"OOB Score: {model.oob_score_:.2f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    print_proba_histogram(y_proba)

    # 5-fold stratified cross-validation to confirm accuracy holds across splits
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced',
    )
    cv_scores = cross_val_score(cv_model, X, y, cv=cv, scoring='accuracy')
    print(f"\n5-Fold CV Accuracy: mean={cv_scores.mean():.3f}, std={cv_scores.std():.3f}")
    print(f"CV fold scores: {np.round(cv_scores, 3)}")

    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_,
    }).sort_values('importance', ascending=False)
    print(f"\nFeature Importance:\n{feature_importance}")

    return model


def save_model(model, feature_columns):
    """Save model and feature columns (no scaler)."""
    model_dir = os.path.dirname(os.path.abspath(__file__))

    model_data = {
        'model': model,
        'feature_columns': feature_columns,
        'model_version': '1.2',
    }

    filepath = os.path.join(model_dir, 'churn_model.pkl')
    joblib.dump(model_data, filepath)
    print(f"\nModel saved to: {filepath}")


def main():
    """Main training pipeline."""
    print("=" * 50)
    print("RETAINR CHURN PREDICTION MODEL - TRAINING")
    print("=" * 50)

    df = load_data()
    validate_dataframe(df)
    X, y, feature_columns = prepare_features(df)
    model = train_model(X, y)
    save_model(model, feature_columns)

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
