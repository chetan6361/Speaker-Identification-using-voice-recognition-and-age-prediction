from flask import Flask, request, render_template, redirect, url_for
import os
import librosa
import joblib
import pickle
import numpy as np
from scipy.io.wavfile import read
from featureextraction import extract_features as extract_speaker_features  # assuming featureextraction is for speaker testing

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'  # Folder to store uploaded audio files
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load models and scalers once to avoid reloading every request
age_model = joblib.load('Age_models/age_prediction_model.pkl')
age_scaler = joblib.load('Age_models/scaler.pkl')
gmm_models = [pickle.load(open(f'Speakers_models/{fname}', 'rb')) for fname in os.listdir('Speakers_models') if fname.endswith('.gmm')]
speakers = [os.path.splitext(fname)[0] for fname in os.listdir('Speakers_models') if fname.endswith('.gmm')]

age_groups = {
    15: 'Teens (15)',
    25: 'Twenties (25)',
    35: 'Thirties (35)',
    45: 'Forties (45)',
    55: 'Fifties (55)',
    65: 'Sixties (65)',
    75: 'Seventies (75)',
    85: 'Eighties (85)',
    95: 'Nineties (95)',
    -1: 'Unknown'
}


def extract_age_features(audio_path, n_mfcc=120):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    features = np.hstack([np.mean(mfccs, axis=1), np.mean(delta_mfccs, axis=1), np.mean(delta2_mfccs, axis=1)])
    return features

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Run age prediction
        features = extract_age_features(filepath)
        scaled_features = age_scaler.transform([features])
        age_prediction = age_model.predict(scaled_features)[0]
        age_group = age_groups.get(age_prediction, 'unknown')

        # Run speaker identification
        sr, audio = read(filepath)
        speaker_features = extract_speaker_features(audio, sr)
        log_likelihood = np.zeros(len(gmm_models))
        for i, gmm in enumerate(gmm_models):
            scores = np.array(gmm.score(speaker_features))
            log_likelihood[i] = scores.sum()
        speaker_prediction = speakers[np.argmax(log_likelihood)]

        return render_template('result.html', age=age_group, speaker=speaker_prediction)

if __name__ == '__main__':
    app.run(debug=True)
