"""Smoke test: import every major dependency listed in requirements.txt."""


def test_core_data_stack():
    import pandas
    import numpy
    import scipy
    import pyarrow

    assert pandas.__version__
    assert numpy.__version__
    assert scipy.__version__
    assert pyarrow.__version__


def test_classical_ml():
    import sklearn
    import xgboost
    import lightgbm
    import shap
    import joblib

    assert sklearn.__version__
    assert xgboost.__version__
    assert lightgbm.__version__
    assert shap.__version__
    assert joblib.__version__


def test_graph_stack():
    import networkx
    import node2vec
    import infomap

    assert networkx.__version__
    assert node2vec
    assert infomap


def test_deep_learning_stack():
    import torch
    import torch_geometric
    import torch_geometric_temporal

    assert torch.__version__
    assert torch_geometric.__version__
    assert torch_geometric_temporal


def test_synth_data_stack():
    import faker
    import geopy

    assert faker.VERSION
    assert geopy.__version__


def test_api_stack():
    import fastapi
    import uvicorn
    import sse_starlette
    import websockets
    import pydantic

    assert fastapi.__version__
    assert uvicorn.__version__
    assert sse_starlette
    assert websockets.__version__
    assert pydantic.VERSION


def test_persistence_stack():
    import sqlalchemy
    import asyncpg
    import redis

    assert sqlalchemy.__version__
    assert asyncpg.__version__
    assert redis.__version__


def test_reporting_stack():
    import reportlab
    import matplotlib
    import seaborn

    assert reportlab.Version
    assert matplotlib.__version__
    assert seaborn.__version__


def test_project_packages():
    import src
    import src.system1_generator
    import src.system2_detection
    import src.system2_detection.shared
    import src.system2_detection.layer1_rules
    import src.system2_detection.layer2_graph
    import src.system2_detection.layer3_supervised
    import src.system2_detection.layer4_anomaly
    import src.system2_detection.layer5_gnn
    import src.system2_detection.post_detection
    import src.system3_evaluation

    assert src
    assert src.system1_generator
    assert src.system2_detection
    assert src.system2_detection.shared
    assert src.system2_detection.layer1_rules
    assert src.system2_detection.layer2_graph
    assert src.system2_detection.layer3_supervised
    assert src.system2_detection.layer4_anomaly
    assert src.system2_detection.layer5_gnn
    assert src.system2_detection.post_detection
    assert src.system3_evaluation
