from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app)

# Load the trained SVM pipeline (TF-IDF + SVC)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "svm_spam_model.pkl")
model = joblib.load(MODEL_PATH)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "No message provided"}), 400

    prediction = model.predict([message])[0]
    label = "spam" if prediction == 1 else "ham"

    # Confidence score via decision function distance from boundary
    try:
        score = float(model.decision_function([message])[0])
        # Normalise to 0-1 range with a sigmoid-like mapping
        import math
        confidence = round(1 / (1 + math.exp(-score)), 4)
    except Exception:
        confidence = None

    return jsonify({
        "message": message,
        "prediction": label,
        "confidence": confidence
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
