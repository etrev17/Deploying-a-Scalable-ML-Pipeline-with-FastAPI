import pytest
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, compute_model_metrics
import math


def test_data_split():
    """
    Assesses whether the data is properly split into training and testing sets.
    """
    project_path = os.getcwd()
    data_path = os.path.join(project_path, "data", "census.csv")
    data = pd.read_csv(data_path)

    train, test = train_test_split(data, test_size=0.2, random_state=0)

    assert not train.empty, "Training dataset is empty."
    assert not test.empty, "Test dataset is empty."
    assert math.isclose(len(test) / len(data), 0.2, rel_tol=0.01), f"Test dataset to precision: {len(test) / len(data):.4f}"


def test_compute_model_metrics():
    """
    Assesses whether compute_model_metrics() correctly calculates
    precision, recall, and fbeta.
    """
    data = np.array([1, 0, 1, 0])
    preds = np.array([1, 1, 0, 0])
    precision, recall, fbeta = compute_model_metrics(data, preds)

    assert precision == 1 / 2, f"Precision: {precision:.3f}. Expected: 0.500."
    assert recall == 1 / 2, f"Recall: {recall:.3f}. Expected: 0.500."
    assert fbeta == 1 / 2, f"F1: {fbeta:.3f}. Expected: 0.500."

def test_model_algorithm():
    """
    Ensures that train_model() returns a RandomForestClassifier.
    """
    x = [[1], [2], [3], [4]]
    y = [1, 0, 1, 0]
    model = train_model(x, y)

    assert isinstance(model, RandomForestClassifier), "Model is not a RandomForestClassifier."