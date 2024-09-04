import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib  # Import joblib for saving the model

# List of CSV files
csv_files = ['dataset-of-60s.csv', 'dataset-of-70s.csv', 'dataset-of-80s.csv', 'dataset-of-90s.csv', 'dataset-of-00s.csv','dataset-of-10s.csv','Merged file.csv']

# Loop through each CSV file
for file in csv_files:
    print(f"Processing {file}...")
    df = pd.read_csv(file)
    df = df.drop(columns=['uri'])

    x = df.iloc[:, 2:-1]
    y = df.iloc[:, -1]

    selected_features = ['instrumentalness', 'danceability', 'loudness', 'valence', 'acousticness', 'key']
    x_1 = df[selected_features]

    x_train, x_test, y_train, y_test = train_test_split(x_1, y, test_size=0.2)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    model2 = RandomForestClassifier()
    model2.fit(x_train, y_train)

    y_pred2 = model2.predict(x_test)
    c = accuracy_score(y_test, y_pred2)

    print(f"Results for {file}:")
    print(f"Random Forest Accuracy: {c}")

    # Save the model and scaler after training
    joblib.dump(model2, 'random_forest_model.pkl')  # Save the trained model
    joblib.dump(scaler, 'scaler.pkl')  # Save the scaler

    print("\n")
