import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.predictor import predictor


def make_png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (120, 80, 200)).save(buf, format="PNG")
    return buf.getvalue()


class FakeModel:
    def __init__(self, p_real):
        self.p_real = p_real

    def predict(self, batch, verbose=0):
        return [[self.p_real]]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(predictor, "load", lambda: None)  # don't need a trained model
    with TestClient(app) as c:
        yield c


def test_health_reports_model_state(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_without_model_returns_503(client, monkeypatch):
    monkeypatch.setattr(predictor, "model", None)
    r = client.post("/api/predict", files={"file": ("a.png", make_png(), "image/png")})
    assert r.status_code == 503


@pytest.mark.parametrize("p_real,label", [(0.93, "Real Images"), (0.07, "AI-Generated Images")])
def test_predict_labels(client, monkeypatch, p_real, label):
    monkeypatch.setattr(predictor, "model", FakeModel(p_real))
    r = client.post("/api/predict", files={"file": ("a.png", make_png(), "image/png")})
    body = r.json()
    assert r.status_code == 200
    assert body["label"] == label
    assert body["confidence"] == pytest.approx(0.93)


def test_rejects_non_image(client, monkeypatch):
    monkeypatch.setattr(predictor, "model", FakeModel(0.5))
    r = client.post("/api/predict", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_rejects_corrupted_image(client, monkeypatch):
    monkeypatch.setattr(predictor, "model", FakeModel(0.5))
    r = client.post("/api/predict", files={"file": ("a.png", b"not an image", "image/png")})
    assert r.status_code == 400
