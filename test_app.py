import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

# Test 1
def test_home_endpoint(client):
    r = client.get("/")
    assert r.status_code == 200

def test_predict_valid(client):
    r = client.post('/predict',
        json={'features': [5.1, 3.5, 1.4, 0.2]})
    assert r.status_code == 200
    assert r.get_json() == {'prediction': 'setosa'}

def test_predict_invalid(client):
    r = client.post('/predict',
        json={'features': [5.1, 3.5, 1.4]})
    assert r.status_code == 400