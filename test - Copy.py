import os
import pickle  # updated import for Python 3
import numpy as np
from scipy.io.wavfile import read
from featureextraction import extract_features  # assuming you have a module named featureextraction.py
import warnings
warnings.filterwarnings("ignore")
import time

# Path to training data
source = "SampleData/"

# Path where training speakers will be saved
modelpath = "Speakers_models/"

gmm_files = [os.path.join(modelpath, fname) for fname in os.listdir(modelpath) if fname.endswith('.gmm')]

# Load the Gaussian Mixture Models
models = [pickle.load(open(fname, 'rb')) for fname in gmm_files]  # 'rb' mode for reading binary files
speakers = [os.path.splitext(os.path.basename(fname))[0] for fname in gmm_files]

while True:
    print("Do you want to Test a Single Audio: Press '1', or Exit: Press '2'?")
    take = int(input().strip())
    
    if take == 1:
        print("Enter the File name from Test Audio Sample Collection:")
        path = input().strip()
        print("Testing Audio:", path)
        sr, audio = read(source + path)
        vector = extract_features(audio, sr)

        log_likelihood = np.zeros(len(models))

        for i in range(len(models)):
            gmm = models[i]  # checking with each model one by one
            scores = np.array(gmm.score(vector))
            log_likelihood[i] = scores.sum()

        winner = np.argmax(log_likelihood)
        print("\tdetected as -", speakers[winner])

        time.sleep(1.0)
    
    # elif take == 0:
    #     error = 0
    #     total_sample = 0.0
        
    #     test_file = "testSamplePath.txt"
    #     with open(test_file, 'r') as file_paths:
    #         # Read the test directory and get the list of test audio files
    #         for path in file_paths:
    #             total_sample += 1.0
    #             path = path.strip()
    #             print("Testing Audio:", path)
    #             sr, audio = read(source + path)
    #             vector = extract_features(audio, sr)

    #             log_likelihood = np.zeros(len(models))

    #             for i in range(len(models)):
    #                 gmm = models[i]  # checking with each model one by one
    #                 scores = np.array(gmm.score(vector))
    #                 log_likelihood[i] = scores.sum()

    #             winner = np.argmax(log_likelihood)
    #             print("\tdetected as -", speakers[winner])

    #             checker_name = os.path.basename(path).split("_")[0]
    #             if speakers[winner] != checker_name:
    #                 error += 1
    #             time.sleep(1.0)

    #     accuracy = ((total_sample - error) / total_sample) * 100
    #     print(f"The Accuracy Percentage for the current testing Performance with MFCC + GMM is: {accuracy:.2f}%")
    
    elif take == 2:
        print("Exiting the testing loop.")
        break

    else:
        print("Invalid input. Please try again.")

print("Hurrah! Speaker identified. Mission Accomplished Successfully.")
