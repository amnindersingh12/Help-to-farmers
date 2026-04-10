import io
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image, UnidentifiedImageError
from flask import Flask, jsonify, redirect, render_template, request
from markupsafe import Markup
from torchvision import transforms
from werkzeug.middleware.proxy_fix import ProxyFix

import config
from my.disease import disease_dic
from my.fertilizer import fertilizer_dic
from my.model import ResNet9

BASE_DIR = Path(__file__).resolve().parent
DISEASE_MODEL_PATH = BASE_DIR / "models" / "plant_disease_model.pth"
CROP_MODEL_PATH = BASE_DIR / "models" / "RandomForest.pkl"
FERTILIZER_DATA_PATH = BASE_DIR / "Data" / "fertilizer.csv"


disease_classes = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy',
]

MODEL_STATE = {
    "startup_errors": [],
    "disease_model": None,
    "crop_model": None,
    "fertilizer_df": None,
    "init_error": None,
}


def setup_logging(app):
    formatter = logging.Formatter(
        "%(asctime)s level=%(levelname)s event=%(name)s message=%(message)s"
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    app.logger.handlers.clear()
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.INFO)


def startup_check():
    errors = []
    for p in (DISEASE_MODEL_PATH, CROP_MODEL_PATH, FERTILIZER_DATA_PATH):
        if not p.exists():
            errors.append(f"Missing required file: {p}")

    if not config.weather_api_key:
        errors.append("WEATHER_API_KEY is not set; crop predictions that require weather will fail.")

    MODEL_STATE["startup_errors"] = errors
    return errors


def ensure_models_loaded():
    if MODEL_STATE["init_error"]:
        raise RuntimeError(MODEL_STATE["init_error"])

    if MODEL_STATE["disease_model"] is None:
        if not DISEASE_MODEL_PATH.exists():
            MODEL_STATE["init_error"] = f"Disease model file not found: {DISEASE_MODEL_PATH}"
            raise RuntimeError(MODEL_STATE["init_error"])

        disease_model = ResNet9(3, len(disease_classes))
        disease_model.load_state_dict(torch.load(str(DISEASE_MODEL_PATH), map_location=torch.device('cpu')))
        disease_model.eval()
        MODEL_STATE["disease_model"] = disease_model

    if MODEL_STATE["crop_model"] is None:
        if not CROP_MODEL_PATH.exists():
            MODEL_STATE["init_error"] = f"Crop model file not found: {CROP_MODEL_PATH}"
            raise RuntimeError(MODEL_STATE["init_error"])

        with open(CROP_MODEL_PATH, "rb") as fh:
            MODEL_STATE["crop_model"] = pickle.load(fh)

    if MODEL_STATE["fertilizer_df"] is None:
        if not FERTILIZER_DATA_PATH.exists():
            MODEL_STATE["init_error"] = f"Fertilizer data file not found: {FERTILIZER_DATA_PATH}"
            raise RuntimeError(MODEL_STATE["init_error"])

        MODEL_STATE["fertilizer_df"] = pd.read_csv(FERTILIZER_DATA_PATH)


def parse_number(form, key, label, caster, min_value=None, max_value=None):
    raw = (form.get(key) or "").strip()
    if raw == "":
        raise ValueError(f"{label} is required.")

    try:
        value = caster(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a valid number.")

    if min_value is not None and value < min_value:
        raise ValueError(f"{label} must be at least {min_value}.")
    if max_value is not None and value > max_value:
        raise ValueError(f"{label} must be at most {max_value}.")

    return value


def weather_fetch(city_name):
    api_key = config.weather_api_key
    if not api_key or not city_name:
        return None

    base_url = "https://api.openweathermap.org/data/2.5/weather"
    try:
        response = requests.get(
            base_url,
            params={"appid": api_key, "q": city_name},
            timeout=8,
        )
        payload = response.json()
    except requests.RequestException:
        return None

    if str(payload.get("cod")) == "404" or "main" not in payload:
        return None

    main = payload["main"]
    temperature = round((main["temp"] - 273.15), 2)
    humidity = main["humidity"]
    return temperature, humidity


def predict_image(img_bytes):
    ensure_models_loaded()
    model = MODEL_STATE["disease_model"]

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)
    yb = model(img_u)
    _, preds = torch.max(yb, dim=1)
    return disease_classes[preds[0].item()]


def create_app():
    app = Flask(__name__)
    app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1, x_prefix=1)
    setup_logging(app)
    startup_errors = startup_check()
    if startup_errors:
        for err in startup_errors:
            app.logger.warning(err)

    @app.route("/health")
    def health():
        return jsonify({
            "ok": True,
            "startup_errors": MODEL_STATE["startup_errors"],
            "model_error": MODEL_STATE["init_error"],
        })

    @app.route('/')
    def home():
        title = 'Help to Farmers- Home'
        return render_template('index.html', title=title)

    @app.route('/crop-recommend')
    def crop_recommend():
        title = 'Help to Farmers- Crop Recommendation'
        return render_template('crop.html', title=title, error=None, form_data={})

    @app.route('/fertilizer')
    def fertilizer_recommendation():
        title = 'Help to Farmers- Fertilizer Suggestion'
        return render_template('fertilizer.html', title=title, error=None, form_data={})

    @app.route('/crop-predict', methods=['POST'])
    def crop_prediction():
        title = 'Help to Farmers- Crop Recommendation'
        form_data = request.form.to_dict(flat=True)

        try:
            n_val = parse_number(request.form, 'nitrogen', 'Nitrogen', int, 0, 500)
            p_val = parse_number(request.form, 'phosphorous', 'Phosphorous', int, 0, 500)
            k_val = parse_number(request.form, 'pottasium', 'Pottasium', int, 0, 500)
            ph = parse_number(request.form, 'ph', 'pH level', float, 0.0, 14.0)
            rainfall = parse_number(request.form, 'rainfall', 'Rainfall', float, 0.0, 10000.0)
            city = (request.form.get('city') or '').strip()
            if not city:
                raise ValueError("City is required.")

            ensure_models_loaded()
            weather = weather_fetch(city)
            if weather is None:
                raise ValueError("Could not fetch weather for the selected city. Please try another city.")

            temperature, humidity = weather
            data = np.array([[n_val, p_val, k_val, temperature, humidity, ph, rainfall]])
            prediction = MODEL_STATE["crop_model"].predict(data)[0]
            app.logger.info("crop_prediction_success city=%s", city)
            return render_template('crop-result.html', prediction=prediction, title=title)
        except ValueError as exc:
            app.logger.info("crop_prediction_validation_failed error=%s", exc)
            return render_template('crop.html', title=title, error=str(exc), form_data=form_data), 400
        except RuntimeError as exc:
            app.logger.error("crop_prediction_runtime_failed error=%s", exc)
            return render_template('crop.html', title=title, error=str(exc), form_data=form_data), 503

    @app.route('/fertilizer-predict', methods=['POST'])
    def fert_recommend():
        title = 'Help to Farmers- Fertilizer Suggestion'
        form_data = request.form.to_dict(flat=True)

        try:
            ensure_models_loaded()
            crop_name = (request.form.get('cropname') or '').strip().lower()
            if not crop_name or crop_name == 'select crop':
                raise ValueError("Please select a crop.")

            n_val = parse_number(request.form, 'nitrogen', 'Nitrogen', int, 0, 500)
            p_val = parse_number(request.form, 'phosphorous', 'Phosphorous', int, 0, 500)
            k_val = parse_number(request.form, 'pottasium', 'Pottasium', int, 0, 500)

            df = MODEL_STATE["fertilizer_df"]
            row = df[df['Crop'].str.lower() == crop_name]
            if row.empty:
                raise ValueError("Selected crop is not supported.")

            nr = int(row['N'].iloc[0])
            pr = int(row['P'].iloc[0])
            kr = int(row['K'].iloc[0])

            n = nr - n_val
            p = pr - p_val
            k = kr - k_val
            temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
            max_value = temp[max(temp.keys())]
            if max_value == "N":
                key = 'NHigh' if n < 0 else 'Nlow'
            elif max_value == "P":
                key = 'PHigh' if p < 0 else 'Plow'
            else:
                key = 'KHigh' if k < 0 else 'Klow'

            recommendation = Markup(str(fertilizer_dic[key]))
            app.logger.info("fertilizer_prediction_success crop=%s", crop_name)
            return render_template('fertilizer-result.html', recommendation=recommendation, title=title)
        except ValueError as exc:
            app.logger.info("fertilizer_prediction_validation_failed error=%s", exc)
            return render_template('fertilizer.html', title=title, error=str(exc), form_data=form_data), 400
        except RuntimeError as exc:
            app.logger.error("fertilizer_prediction_runtime_failed error=%s", exc)
            return render_template('fertilizer.html', title=title, error=str(exc), form_data=form_data), 503

    @app.route('/disease-predict', methods=['GET', 'POST'])
    def disease_prediction():
        title = 'Help to Farmers- Disease Detection'

        if request.method == 'POST':
            file = request.files.get('file')
            if not file or not file.filename:
                return render_template('disease.html', title=title, error="Please upload an image file."), 400

            if not file.mimetype.startswith("image/"):
                return render_template('disease.html', title=title, error="Uploaded file must be an image."), 400

            try:
                img = file.read()
                if not img:
                    raise ValueError("Uploaded image is empty.")

                prediction_key = predict_image(img)
                prediction = Markup(str(disease_dic[prediction_key]))
                app.logger.info("disease_prediction_success")
                return render_template('disease-result.html', prediction=prediction, title=title)
            except (UnidentifiedImageError, ValueError):
                return render_template('disease.html', title=title, error="Could not read the image. Please upload a valid image file."), 400
            except RuntimeError as exc:
                app.logger.error("disease_prediction_runtime_failed error=%s", exc)
                return render_template('disease.html', title=title, error=str(exc)), 503
            except Exception as exc:
                app.logger.exception("disease_prediction_unexpected_error")
                return render_template('disease.html', title=title, error=f"Prediction failed: {exc}"), 500

        return render_template('disease.html', title=title, error=None)

    return app


app = create_app()
application = app


if __name__ == '__main__':
    app.run(debug=False)
