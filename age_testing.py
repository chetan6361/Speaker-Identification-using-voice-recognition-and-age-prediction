import numpy as np
import librosa
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

def load_model(model_file):
    return joblib.load(model_file)

def load_scaler(scaler_file):
    return joblib.load(scaler_file)

def preprocess_features(features, scaler):
    return scaler.transform([features])

def predict_age(model, features):
    return model.predict(features)[0]

def main(audio_path, model_file, scaler_file):
    # Extract features from audio file
    print("Extracting features...")
    features = extract_features(audio_path)
    
    # Load the model and scaler
    print("Loading model and scaler...")
    model = load_model(model_file)
    scaler = load_scaler(scaler_file)
    
    # Preprocess features
    features = preprocess_features(features, scaler)
    
    # Make a prediction
    print("Making prediction...")
    age_prediction = predict_age(model, features)
    
    # Output the result
    print(f"Predicted Age Class: {age_prediction}")

if __name__ == "__main__":
    audio_path = r"SampleData\Chetan_4.wav"
    model_file = 'Age_models/age_prediction_model.pkl'
    scaler_file = 'Age_models/scaler.pkl'
    
    main(audio_path, model_file, scaler_file)
