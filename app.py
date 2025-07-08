from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('C:\\Users\\HP\Downloads\\customer_churn_project\\customer_churn_project\\churn_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    prediction = model.predict([np.array(features)])
    probabilities = model.predict_proba([np.array(features)])
    churn_prob = round(probabilities[0][1] * 100, 2)  # Probability for 'Yes'
    churn_label = 'Yes' if prediction[0] == 1 else 'No'
    result_text = f"Churn Prediction: {churn_label} (Probability: {churn_prob}%)"
    return render_template('index.html', prediction_text=result_text)

if __name__ == "__main__":
    app.run(debug=True)
