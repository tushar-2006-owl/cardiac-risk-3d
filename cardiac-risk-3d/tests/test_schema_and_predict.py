"""Schema, probability range, risk thresholds, and vessel mesh mapping."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ml.config import PROJECT_ROOT, RISK_THRESHOLDS, VESSEL_MESH_KEYS, risk_level


@pytest.fixture(scope="module")
def client():
    from backend.main import app

    return TestClient(app)


def test_health(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_schema_groups(client: TestClient):
    r = client.get("/schema")
    assert r.status_code == 200
    feats = r.json()["features"]
    assert feats
    groups = {f["group"] for f in feats}
    for g in ("Demographics", "Clinical exam", "ECG", "Laboratory", "Echo"):
        assert g in groups
    names = {f["name"] for f in feats}
    assert "LAD" not in names and "Cath" not in names


def test_predict_probabilities_in_unit_interval(client: TestClient):
    schema = client.get("/schema").json()["features"]
    payload = {item["name"]: item["default"] for item in schema}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0.0 <= body["cad"]["probability"] <= 1.0
    for vessel, pred in body["vessels"].items():
        assert vessel in VESSEL_MESH_KEYS
        assert pred["mesh_key"] == VESSEL_MESH_KEYS[vessel]
        assert 0.0 <= pred["probability"] <= 1.0
        assert pred["risk_level"] in {"low", "medium", "high"}
    assert "educational" in body["disclaimer"].lower() or "decision support" in body["disclaimer"].lower()


def test_unknown_feature_is_422(client: TestClient):
    r = client.post("/predict", json={"not_a_real_column": 1})
    assert r.status_code == 422


def test_risk_level_thresholds():
    assert risk_level(0.0) == "low"
    assert risk_level(RISK_THRESHOLDS["low"] - 1e-9) == "low"
    assert risk_level(0.5) == "medium"
    assert risk_level(RISK_THRESHOLDS["high"] + 1e-9) == "high"


def test_vessel_config_mesh_keys_match_frontend():
    ts = (PROJECT_ROOT / "frontend" / "src" / "config" / "vessels.ts").read_text(encoding="utf-8")
    for vessel, mesh in VESSEL_MESH_KEYS.items():
        assert re.search(rf"{vessel}\s*:\s*\{{", ts), vessel
        assert f'meshKey: "{mesh}"' in ts or f"meshKey: '{mesh}'" in ts
    assert '"LAD"' in ts or "LAD:" in ts
    assert "LCX" in ts and "RCA" in ts
