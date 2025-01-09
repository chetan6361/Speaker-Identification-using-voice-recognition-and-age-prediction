import os
import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

def extract_features(audio_path, n_mfcc=120):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    
    # Aggregate features by mean
    mfccs_mean = np.mean(mfccs, axis=1)
    delta_mfccs_mean = np.mean(delta_mfccs, axis=1)
    delta2_mfccs_mean = np.mean(delta2_mfccs, axis=1)
    
    # Concatenate all features into a single vector
    features = np.hstack([mfccs_mean, delta_mfccs_mean, delta2_mfccs_mean])
    return features

def map_age_to_numeric(age_str):
    age_map = {
        'teens': 15,
        'twenties': 25,
        'thirties': 35,
        'fourties': 45,
        'fifties': 55,
        'sixties': 65,
        'seventies': 75,
        'eighties': 85,
        'nineties': 95,
        'unknown': -1  # Handle unknown age or outliers
    }
    return age_map.get(age_str.lower(), -1)  # Default to -1 if age not found

def process_dataset(audio_dir, csv_path, output_file):
    print("Loading CSV file...")
    df = pd.read_csv(csv_path)
    features = []
    labels = []
    
    print("Extracting features...")
    for _, row in df.iterrows():
        audio_path = os.path.join(audio_dir, row['filename'])
        if os.path.isfile(audio_path):
            mfccs = extract_features(audio_path)
            features.append(mfccs)
            age_numeric = map_age_to_numeric(row['age'])
            labels.append(age_numeric)
        else:
            print(f"File not found: {audio_path}")
    
    features = np.array(features)
    labels = np.array(labels)
    
    if features.size == 0:
        raise ValueError("No features extracted. Please check your audio files and extraction function.")
    
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    features_df = pd.DataFrame(features)
    features_df['age'] = labels
    features_df.to_csv(output_file, index=False)
    
    # Save the scaler for future use
    scaler_file = 'Age_models/scaler.pkl'
    joblib.dump(scaler, scaler_file)
    print(f"Scaler saved to {scaler_file}")
    
    print(f"Features saved to {output_file}")

def train_age_prediction_model(processed_csv_path):
    # Load the processed data
    print("Loading processed features...")
    df = pd.read_csv(processed_csv_path)
    
    # Separate features and labels
    X = df.drop('age', axis=1)
    y = df['age']
    
    # Get unique class labels
    unique_labels = sorted(y.unique())
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    print("Training the model...")
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    print("Making predictions...")
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    print("Evaluation results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    
    # Print classification report
    print(classification_report(y_test, y_pred, labels=unique_labels,
                                target_names=[f'Class {label}' for label in unique_labels]))
    
    # Save the model for future use
    model_file = 'Age_models/age_prediction_model.pkl'
    joblib.dump(model, model_file)
    print(f"Model saved to {model_file}")

if __name__ == "__main__":
    audio_dir = 'cv-valid-test'
    csv_path = 'cv-valid-test.csv'
    processed_csv_path = 'processed_features.csv'
    
    process_dataset(audio_dir, csv_path, processed_csv_path)
    train_age_prediction_model(processed_csv_path)
