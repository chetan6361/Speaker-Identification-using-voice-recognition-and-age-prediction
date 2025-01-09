import os
import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import StandardScaler

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
        'forties': 45,
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
    
    print(f"Features saved to {output_file}")

if __name__ == "__main__":
    audio_dir = 'cv-valid-test'
    csv_path = 'cv-valid-test.csv'
    output_file = 'processed_features.csv'
    
    process_dataset(audio_dir, csv_path, output_file)
