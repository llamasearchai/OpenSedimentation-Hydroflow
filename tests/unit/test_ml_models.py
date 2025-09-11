"""Unit tests for machine learning models (subset)."""

import numpy as np
import pandas as pd

from hydroflow.ml.models import SedimentationPredictor, FlowPatternPredictor


def test_feature_preparation():
    predictor = SedimentationPredictor()
    flow_data = pd.DataFrame(
        {
            "velocity": np.random.uniform(0.5, 2.0, 100),
            "depth": np.random.uniform(1.0, 5.0, 100),
            "discharge": np.random.uniform(10, 100, 100),
        }
    )
    precipitation = pd.DataFrame({"daily": np.random.uniform(0, 50, 30), "intensity": np.random.uniform(0, 10, 30)})
    land_use = pd.DataFrame({"type": ["urban", "agricultural", "forest"], "area": [100, 200, 300]})
    bathymetry = np.random.randn(50, 50) * 2 - 5
    features = predictor.prepare_features(flow_data, precipitation, land_use, bathymetry)
    assert features.shape[0] == 1
    assert features.shape[1] > 10


def test_training_and_prediction():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 20))
    y = X[:, :5].sum(axis=1) + rng.standard_normal(200) * 0.1
    predictor = SedimentationPredictor(model_type="rf")
    metrics = predictor.train(X, y, validation_split=0.2)
    assert "train_r2" in metrics
    preds = predictor.predict(X[:10])
    assert preds.shape[0] == 10


def test_flow_pattern_predictor_forward():
    model = FlowPatternPredictor(input_size=10, hidden_sizes=[32, 16], output_size=5)
    X = np.random.randn(5, 10)
    out = model.forward(X)
    assert out.shape == (5, 5)


