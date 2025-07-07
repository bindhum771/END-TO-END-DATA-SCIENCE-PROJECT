from flask import Flask, render_template, request
import joblib
import numpy as np
from datasets import load_dataset

ds = load_dataset("scikit-learn/iris")

app = Flask(__name__)
model = joblib.load('model/iris_model.pkl')


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        features = [float(request.form[f]) for f in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
        prediction = model.predict([features])[0]
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)