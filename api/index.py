import os
import sys
import joblib
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

# Ensure clean UTF-8 console output on Windows environments
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

app = Flask(__name__)


def find_model_path():
    """
    Locates iris_model.pkl across local development and Vercel serverless runtime environments.
    """
    candidate_paths = [
        # 1. Project root directory (standard in Vercel function runtime)
        os.path.join(os.getcwd(), 'iris_model.pkl'),
        # 2. Parent directory of this script (/api -> project root)
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'iris_model.pkl'),
        # 3. Same directory as this script (/api)
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'iris_model.pkl'),
    ]

    for path in candidate_paths:
        if os.path.isfile(path):
            return path

    return candidate_paths[0]


MODEL_PATH = find_model_path()

# Load the pre-trained model once at module initialization
# In serverless environments, this remains cached across warm function executions
model = None
try:
    if os.path.isfile(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"[OK] Successfully loaded model from: {MODEL_PATH}")
    else:
        print(f"[WARN] Model file not found at: {MODEL_PATH}")
except Exception as err:
    print(f"[WARN] joblib.load failed ({err}), attempting fallback...")
    try:
        import pickle
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print(f"[OK] Successfully loaded model via pickle from: {MODEL_PATH}")
    except Exception as pickle_err:
        print(f"[ERROR] Model load failed: {pickle_err}")


# Species metadata dictionary
SPECIES_INFO = {
    0: {
        "species": "Setosa",
        "scientific_name": "Iris setosa",
        "description": "Characterized by smaller, delicate petals and wide sepals. Common in northern and arctic regions.",
        "icon": "🌸",
        "color": "#10b981",
        "gradient": "linear-gradient(135deg, #10b981, #059669)"
    },
    1: {
        "species": "Versicolor",
        "scientific_name": "Iris versicolor",
        "description": "Also known as the Blue Flag iris. Features medium-sized petals with distinct, elegant veining.",
        "icon": "🌿",
        "color": "#3b82f6",
        "gradient": "linear-gradient(135deg, #3b82f6, #1d4ed8)"
    },
    2: {
        "species": "Virginica",
        "scientific_name": "Iris virginica",
        "description": "Also known as the Virginia iris. Features large, vibrant blossoms with elongated petals.",
        "icon": "🌺",
        "color": "#8b5cf6",
        "gradient": "linear-gradient(135deg, #8b5cf6, #6d28d9)"
    }
}


@app.after_request
def add_cors_headers(response):
    """Adds standard CORS headers for cross-origin API compatibility."""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response


@app.route('/api/health', methods=['GET'])
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint to verify serverless function status and model loading."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "classes": [SPECIES_INFO[c]["species"] for c in sorted(SPECIES_INFO.keys())]
    })


@app.route('/api/predict', methods=['POST', 'OPTIONS'])
@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    """
    Serverless prediction endpoint:
    Receives JSON with sepal_length, sepal_width, petal_length, petal_width.
    Returns predicted Iris species and confidence metrics.
    """
    if request.method == 'OPTIONS':
        return ('', 204)

    # Verify model is available
    if model is None:
        return jsonify({
            "success": False,
            "error": "The machine learning model could not be loaded on the server. Please verify iris_model.pkl is deployed."
        }), 500

    # Parse incoming payload (supports JSON and Form data)
    payload = request.get_json(silent=True)
    if not payload and request.form:
        payload = request.form

    if not payload:
        return jsonify({
            "success": False,
            "error": "Missing input data. Please provide sepal_length, sepal_width, petal_length, and petal_width."
        }), 400

    # Extract & validate the 4 measurements
    try:
        sl = float(str(payload.get('sepal_length', '')).strip())
        sw = float(str(payload.get('sepal_width', '')).strip())
        pl = float(str(payload.get('petal_length', '')).strip())
        pw = float(str(payload.get('petal_width', '')).strip())
    except (ValueError, TypeError):
        return jsonify({
            "success": False,
            "error": "All 4 measurements must be valid numeric values."
        }), 400

    # Physical dimension validation
    for val, name in [
        (sl, "Sepal Length"),
        (sw, "Sepal Width"),
        (pl, "Petal Length"),
        (pw, "Petal Width"),
    ]:
        if val <= 0:
            return jsonify({
                "success": False,
                "error": f"{name} must be greater than 0."
            }), 400
        if val > 30:
            return jsonify({
                "success": False,
                "error": f"{name} is unrealistically large (> 30 cm)."
            }), 400

    # Format 2D array input [1, 4] for scikit-learn
    features = np.array([[sl, sw, pl, pw]], dtype=float)

    # Perform model inference
    try:
        raw_pred = model.predict(features)[0]
        class_id = int(raw_pred)
    except Exception as pred_err:
        return jsonify({
            "success": False,
            "error": f"Prediction inference error: {str(pred_err)}"
        }), 500

    # Retrieve species metadata
    species_data = SPECIES_INFO.get(class_id, {
        "species": f"Class {class_id}",
        "scientific_name": "Unknown species",
        "description": "Model predicted an unrecognized class index.",
        "icon": "🌱",
        "color": "#64748b",
        "gradient": "linear-gradient(135deg, #64748b, #475569)"
    })

    # Calculate probabilities if supported by the model
    confidence = None
    probabilities = {}
    if hasattr(model, 'predict_proba'):
        try:
            probs = model.predict_proba(features)[0]
            confidence = round(float(probs[class_id]) * 100, 1)
            for idx, prob in enumerate(probs):
                name = SPECIES_INFO.get(idx, {}).get("species", f"Class {idx}")
                probabilities[name] = round(float(prob) * 100, 1)
        except Exception:
            pass

    return jsonify({
        "success": True,
        "result": {
            "species": species_data["species"],
            "scientific_name": species_data["scientific_name"],
            "description": species_data["description"],
            "icon": species_data["icon"],
            "color": species_data["color"],
            "gradient": species_data["gradient"],
            "confidence": confidence,
            "probabilities": probabilities,
            "inputs": {
                "sepal_length": sl,
                "sepal_width": sw,
                "petal_length": pl,
                "petal_width": pw
            }
        }
    })


# Static file serving fallback for local development without Vercel CLI
PUBLIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'public')


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    """Serves static frontend files when testing locally."""
    if path and os.path.isfile(os.path.join(PUBLIC_DIR, path)):
        return send_from_directory(PUBLIC_DIR, path)

    index_file = os.path.join(PUBLIC_DIR, 'index.html')
    if os.path.isfile(index_file):
        return send_from_directory(PUBLIC_DIR, 'index.html')

    return jsonify({
        "message": "Iris Flower Classification Serverless API is running.",
        "endpoints": ["/api/predict", "/api/health"]
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"[OK] Local development server running at: http://127.0.0.1:{port}")
    app.run(host='127.0.0.1', port=port, debug=True)
