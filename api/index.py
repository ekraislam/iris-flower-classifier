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
    Locates iris_model.pkl across local development, AWS Lambda, and Vercel serverless environments.
    """
    candidate_paths = [
        # Inside /api directory
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'iris_model.pkl'),
        # Root directory from parent of /api
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'iris_model.pkl'),
        # Current working directory
        os.path.join(os.getcwd(), 'iris_model.pkl'),
        os.path.join(os.getcwd(), 'api', 'iris_model.pkl'),
        # AWS Lambda standard task root
        '/var/task/iris_model.pkl',
        '/var/task/api/iris_model.pkl',
    ]

    for path in candidate_paths:
        if os.path.isfile(path):
            return path

    return candidate_paths[0]


MODEL_PATH = find_model_path()

# Load the pre-trained model once at module initialization
model = None
model_error = None

try:
    if os.path.isfile(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"[OK] Successfully loaded model from: {MODEL_PATH}")
    else:
        model_error = f"Model file not found at: {MODEL_PATH}"
        print(f"[WARN] {model_error}")
except Exception as err:
    print(f"[WARN] joblib.load failed ({err}), attempting fallback...")
    try:
        import pickle
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print(f"[OK] Successfully loaded model via pickle from: {MODEL_PATH}")
    except Exception as pickle_err:
        model_error = f"joblib: {err}; pickle: {pickle_err}"
        print(f"[ERROR] Model load failed: {model_error}")


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
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response


def do_prediction():
    """Performs inference with the loaded model."""
    if model is None:
        return jsonify({
            "success": False,
            "error": f"The model is not loaded on the server. Details: {model_error or 'File not found'}"
        }), 500

    payload = request.get_json(silent=True)
    if not payload and request.form:
        payload = request.form

    if not payload:
        return jsonify({
            "success": False,
            "error": "Missing input data. Please provide sepal_length, sepal_width, petal_length, and petal_width."
        }), 400

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

    for val, name in [
        (sl, "Sepal Length"),
        (sw, "Sepal Width"),
        (pl, "Petal Length"),
        (pw, "Petal Width"),
    ]:
        if val <= 0:
            return jsonify({"success": False, "error": f"{name} must be greater than 0."}), 400
        if val > 30:
            return jsonify({"success": False, "error": f"{name} is unrealistically large (> 30 cm)."}), 400

    features = np.array([[sl, sw, pl, pw]], dtype=float)

    try:
        raw_pred = model.predict(features)[0]
        class_id = int(raw_pred)
    except Exception as pred_err:
        return jsonify({
            "success": False,
            "error": f"Prediction inference error: {str(pred_err)}"
        }), 500

    species_data = SPECIES_INFO.get(class_id, {
        "species": f"Class {class_id}",
        "scientific_name": "Unknown species",
        "description": "Model predicted an unrecognized class index.",
        "icon": "🌱",
        "color": "#64748b",
        "gradient": "linear-gradient(135deg, #64748b, #475569)"
    })

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


# Universal handler accepting all HTTP methods and paths
# This guarantees Vercel rewrites never fail with 405 Method Not Allowed
PUBLIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'public')


@app.route('/', defaults={'path': ''}, methods=['GET', 'POST', 'OPTIONS'])
@app.route('/<path:path>', methods=['GET', 'POST', 'OPTIONS'])
def universal_handler(path=''):
    if request.method == 'OPTIONS':
        return ('', 204)

    # Any POST request triggers prediction
    if request.method == 'POST':
        return do_prediction()

    # Health check endpoints
    if 'health' in path or request.args.get('health'):
        return jsonify({
            "status": "healthy",
            "model_loaded": model is not None,
            "model_path": MODEL_PATH,
            "model_error": model_error
        })

    # Serve static assets when running in local development mode
    if path and os.path.isfile(os.path.join(PUBLIC_DIR, path)):
        return send_from_directory(PUBLIC_DIR, path)

    index_file = os.path.join(PUBLIC_DIR, 'index.html')
    if os.path.isfile(index_file):
        return send_from_directory(PUBLIC_DIR, 'index.html')

    return jsonify({
        "status": "ready",
        "model_loaded": model is not None,
        "endpoints": ["/api/predict", "/api/health"]
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"[OK] Local server running at http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)
