from flask import Flask, request, render_template
import pickle
import csv
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import flask.app

# Create Flask app instance


app = Flask(__name__)

# Load the model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# This shows the form page
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# This handles form submission
@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["text"]
    vect_text = vectorizer.transform([text])
    prediction = model.predict(vect_text)[0]
    result = "Positive" if prediction == 1 else "Negative"
    return render_template("index.html", prediction=result, input_text=text)


if __name__ == "__main__":
    app.run(debug=True)
