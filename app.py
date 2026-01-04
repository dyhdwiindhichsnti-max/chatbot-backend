import json
import pickle
import random

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load dataset
with open("dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

CONFIDENCE_THRESHOLD = 0.15

def get_response(user_input):
    user_input = user_input.lower()

    X = vectorizer.transform([user_input])
    probs = model.predict_proba(X)[0]
    max_prob = max(probs)

    if max_prob < CONFIDENCE_THRESHOLD:
        # fallback
        for intent in data["intents"]:
            if intent["tag"] == "fallback":
                return random.choice(intent["responses"])

    tag_index = probs.argmax()
    tag = label_encoder.inverse_transform([tag_index])[0]

    for intent in data["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "Mohon maaf, kami belum memahami pesan Anda 😥"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")
    reply = get_response(user_message)
    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(debug=True)
