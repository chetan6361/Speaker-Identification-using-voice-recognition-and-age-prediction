import pickle
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture
from featureextraction import extract_features
import warnings
warnings.filterwarnings("ignore")

# Path to training data
source = "trainingData/"

# Path where training speakers will be saved
dest = "Speakers_models/"
train_file = "trainingDataPath.txt"
file_paths = open(train_file, 'r')

count = 1
# Extracting features for each speaker (5 files per speaker)
features = np.asarray(())

for path in file_paths:
    path = path.strip()
    print(path)
    
    # Read the audio
    sr, audio = read(source + path)
    
    # Extract 40-dimensional MFCC & delta MFCC features
    vector = extract_features(audio, sr)
    
    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector))
    

    if count == 16:
        gmm = GaussianMixture(n_components=16, max_iter=200, covariance_type='diag', n_init=3)
        gmm.fit(features)
        
        
        # Dumping the trained Gaussian model
        picklefile = path.split("-")[0] + ".gmm"
        with open(dest + picklefile, 'wb') as f:
            pickle.dump(gmm, f)
        print(f'+ Modeling completed for speaker: {picklefile} with data point = {features.shape}')
        features = np.asarray(())
        count = 0
    count += 1

file_paths.close()
