"""Local smoke test for the 1.3 API surface, run before deploying to Render.

Exercises the three cases most likely to break: a healthy account, a silent one
using the empty-window convention, and a batch request that omits the interaction
features entirely so the defaults have to supply them.
"""

import json

import app as api

HEALTHY = {
    'account_id': 'SMOKE_HEALTHY',
    'login_frequency': 52,
    'feature_usage_score': 88,
    'support_ticket_count': 1,
    'nps_score': 9,
    'contract_value': 75000,
    'duration_months': 30,
    'total_interactions_30d': 24,
    'negative_sentiment_ratio': 4.2,
    'avg_interaction_duration': 41.5,
    'days_since_last_interaction': 2,
    'support_interaction_ratio': 8.3,
}

SILENT = {
    'account_id': 'SMOKE_SILENT',
    'login_frequency': 3,
    'feature_usage_score': 12,
    'support_ticket_count': 9,
    'nps_score': 2,
    'contract_value': 9000,
    'duration_months': 4,
    'total_interactions_30d': 0,
    'negative_sentiment_ratio': 0,
    'avg_interaction_duration': 0,
    'days_since_last_interaction': 90,
    'support_interaction_ratio': 0,
}

SOURING = {
    'account_id': 'SMOKE_SOURING',
    'login_frequency': 28,
    'feature_usage_score': 61,
    'support_ticket_count': 7,
    'nps_score': 4,
    'contract_value': 45000,
    'duration_months': 20,
    'total_interactions_30d': 11,
    'negative_sentiment_ratio': 63.6,
    'avg_interaction_duration': 8.0,
    'days_since_last_interaction': 41,
    'support_interaction_ratio': 72.7,
}


def main():
    client = api.app.test_client()

    print(f"model_version: {api.model_data['model_version']}")
    print(f"features: {len(api.model_data['feature_columns'])}")

    for payload in (HEALTHY, SILENT, SOURING):
        res = client.post('/predict', json=payload)
        body = res.get_json()
        assert res.status_code == 200, body
        assert body['success'], body
        print(f"\n{payload['account_id']}: {body['churn_score']} {body['risk_level']}")
        print(f"  factors: {body['contributing_factors']}")

    # Batch, with the interaction features omitted so FEATURE_DEFAULTS fills them.
    bare = {k: v for k, v in SOURING.items()
            if k in ('account_id', 'login_frequency', 'feature_usage_score',
                     'support_ticket_count', 'nps_score', 'contract_value',
                     'duration_months')}
    res = client.post('/predict/batch', json={'accounts': [bare]})
    body = res.get_json()
    assert res.status_code == 200, body
    print(f"\nbatch with defaults: {json.dumps(body['predictions'][0], indent=2)}")

    res = client.get('/health')
    print(f"\nhealth: {res.get_json()}")


if __name__ == '__main__':
    main()
