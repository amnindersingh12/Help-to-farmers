import io

import pandas as pd
import pytest

import app as app_module


class DummyCropModel:
    def predict(self, data):
        return ["rice"]


@pytest.fixture
def client(monkeypatch):
    app_module.app.config.update(TESTING=True)

    # Avoid loading large model artifacts in tests.
    monkeypatch.setattr(app_module, "ensure_models_loaded", lambda: None)

    app_module.MODEL_STATE["crop_model"] = DummyCropModel()
    app_module.MODEL_STATE["fertilizer_df"] = pd.DataFrame(
        [{"Crop": "rice", "N": 80, "P": 40, "K": 40}]
    )

    with app_module.app.test_client() as test_client:
        yield test_client


def test_crop_predict_validation_error(client):
    response = client.post(
        "/crop-predict",
        data={
            "nitrogen": "abc",
            "phosphorous": "40",
            "pottasium": "50",
            "ph": "6.5",
            "rainfall": "120",
            "city": "Delhi",
        },
    )
    assert response.status_code == 400
    assert b"Nitrogen must be a valid number" in response.data


def test_crop_predict_success(client, monkeypatch):
    monkeypatch.setattr(app_module, "weather_fetch", lambda city: (27.1, 56))

    response = client.post(
        "/crop-predict",
        data={
            "nitrogen": "90",
            "phosphorous": "40",
            "pottasium": "40",
            "ph": "6.5",
            "rainfall": "120",
            "city": "Delhi",
        },
    )
    assert response.status_code == 200
    assert b"rice" in response.data


def test_fertilizer_predict_validation_error(client):
    response = client.post(
        "/fertilizer-predict",
        data={
            "cropname": "",
            "nitrogen": "80",
            "phosphorous": "40",
            "pottasium": "40",
        },
    )
    assert response.status_code == 400
    assert b"Please select a crop" in response.data


def test_fertilizer_predict_success(client):
    response = client.post(
        "/fertilizer-predict",
        data={
            "cropname": "rice",
            "nitrogen": "60",
            "phosphorous": "40",
            "pottasium": "40",
        },
    )
    assert response.status_code == 200
    assert b"Fertilizer" in response.data


def test_disease_predict_missing_file(client):
    response = client.post("/disease-predict", data={})
    assert response.status_code == 400
    assert b"Please upload an image file" in response.data


def test_disease_predict_success(client, monkeypatch):
    monkeypatch.setattr(app_module, "predict_image", lambda img: "Tomato___healthy")

    response = client.post(
        "/disease-predict",
        data={"file": (io.BytesIO(b"fake-image-bytes"), "leaf.jpg", "image/jpeg")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert b"Tomato" in response.data
