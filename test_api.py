from fastapi.testclient import TestClient
from api import app 

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "threshold" in data
    assert "n_features" in data

def test_predict_ok():
    payload = {
    "fixed acidity": 7.4,
    "volatile acidity": 0.7,
    "citric acid": 0.0,
    "residual sugar": 1.9,
    "chlorides": 0.076,
    "free sulfur dioxide": 11,
    "total sulfur dioxide": 34,
    "density": 0.9978,
    "ph": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4
    } 

    r = client.post("/predict", json=payload)
    assert r.status_code == 200, r.json()

    data = r.json()
    assert "good_quality" in data
    assert "probability" in data
    assert isinstance(data["good_quality"], int)
    assert isinstance(data["probability"], float)

def test_predict_missing_feature():
    bad_payload = {
        "fixed_acidity": 7.4  # incomplete on purpose
    }

    r = client.post("/predict", json=bad_payload)
    assert r.status_code == 422
