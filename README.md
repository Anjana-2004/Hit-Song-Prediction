# Hit-Song-Prediction
A Random Forest model predicts whether a song will be a hit or a flop by analyzing features like instrumentalness, danceability, loudness, valence, acousticness, and key. By combining predictions from multiple decision trees, it reduces overfitting and improves accuracy, leading to more reliable predictions.

Based on the extracted contents of your "Hit Song Prediction" project, here's a possible `README.md` file:

## Project Structure

- **`app.py`**: Main Flask application that handles the web interface and prediction logic.
- **`final_model.py`**: Contains the TensorFlow model used to make predictions using RandomForestClassifier Model.
- **`models.py`**: Includes the code to load and handle the different models used for prediction.
- **`random_forest_model.pkl`**: Pre-trained Random Forest model used for prediction.
- **`scaler.pkl`**: Scaler used to standardize the input features.
- **CSV Files**: Datasets from different decades used for training and evaluation.
  - `dataset-of-60s.csv`
  - `dataset-of-70s.csv`
  - `dataset-of-80s.csv`
  - `dataset-of-90s.csv`
  - `dataset-of-00s.csv`
  - `dataset-of-10s.csv`
  - `Merged file.csv`: Combined dataset across all decades.
- **Templates**: HTML files for the web interface.
  - `index.html`: Form to input song parameters.
  - `result.html`: Displays the prediction result.

## Installation

1. Clone the repository.
2. Install the required dependencies using:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure you have the necessary models (`random_forest_model.pkl` and `scaler.pkl`) in the project directory.
4. Run the Flask app:
   ```bash
   python app.py
   ```

## Usage

1. Open your browser and go to `http://127.0.0.1:5000/`.
2. Input the song parameters:
   - `instrumentalness`
   - `danceability`
   - `loudness`
   - `valence`
   - `acousticness`
   - `key`
3. Submit the form to get the prediction (HIT or FLOP).

## Datasets

The datasets used for model training consist of various attributes of songs from different decades (60s to 10s). They are included in the project for further analysis or retraining.
---
