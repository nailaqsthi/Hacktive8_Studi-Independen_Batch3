import numpy as np
from flask import Flask, render_template, request, jsonify
import pickle
import math

app = Flask(__name__)
model = pickle.load(open("model_bsmote.pkl", "rb"))

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    feature = [np.array(float_features)]
    prediction = model.predict(feature)
    output = {0: 'tidak terkena gagal jantung', 1:'terkena gagal jantung'}
    #output = round(prediction[0],2)
    return render_template("index.html", prediction_text = "Pasien {}".format(output[prediction[0]]))

if __name__ == "__main__":
    app.run(debug=True)