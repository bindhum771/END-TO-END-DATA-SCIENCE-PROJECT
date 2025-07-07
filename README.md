# 🌼 Full Data Science Project with Flask

This project shows an end-to-end data science workflow: from data exploration to training a machine learning model, and deploying it as a web app using Flask.

## 📦 Project Structure
```
full_data_science_project_flask/
├── data_pipeline.ipynb        # Data exploration and visualization
├── train_model.py            # Train and save the ML model
├── app.py                    # Flask app to serve predictions
├── model/
│   └── iris_model.pkl        # Saved trained model
├── templates/
│   └── index.html            # HTML template
├── static/
│   └── style.css             # CSS for styling
└── requirements.txt          # Python dependencies
```

## ⚙️ How to run

1️⃣ **Clone or unzip the project**

2️⃣ **Install dependencies**  
```bash
pip install -r requirements.txt
```

3️⃣ **Train the model** (this will create `model/iris_model.pkl`)
```bash
python train_model.py
```

4️⃣ **Start the Flask app**
```bash
python app.py
```

5️⃣ **Open in browser:**  
Go to [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🧪 **What this project does**
- Loads Iris dataset (from sklearn)
- Data exploration and visualization in `data_pipeline.ipynb`
- Trains a RandomForestClassifier
- Saves the model with `joblib`
- Flask app serves a form to input features
- Predicts and displays the predicted Iris species

---

## ✅ **Requirements**
- Python 3.7+
- Flask
- scikit-learn
- pandas
- numpy
- joblib

Install all using:
```bash
pip install -r requirements.txt
```

Enjoy exploring and extending this project! 🚀