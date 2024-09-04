
from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the form
    instrumentalness = float(request.form['instrumentalness'])
    danceability = float(request.form['danceability'])
    loudness = float(request.form['loudness'])
    valence = float(request.form['valence'])
    acousticness = float(request.form['acousticness'])
    key = float(request.form['key'])

    # Prepare the data for prediction
    input_data = np.array([[instrumentalness, danceability, loudness, valence, acousticness, key]])
    input_data = scaler.transform(input_data)

    # Make a prediction
    prediction = model.predict(input_data)[0]

    # Convert prediction to HIT or FLOP
    result = "HIT" if prediction == 1 else "FLOP"

    return render_template('result.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
