import Preprocess as pp
from keras.models import load_model
import numpy as np
import os
import tensorflow as tf

# Path to the saved model
base_path = 'Saved Processed Data and Models/court_trial (frame_length=0.025 hop_length=0.01 num_samples= 700)'
model_path = os.path.join(base_path, 'Bidirectional LSTM 0.62 Lss 79.17 Acc.h5') # best generalization

# Load the saved model
model = load_model(model_path)

print("For best results, your sample should be 20-30 seconds long.")
while True:
    print("Enter sample path (or type 'exit' to quit):")
    sample_path = input()

    if not os.path.exists(sample_path):
        print("Please Make sure the path is correct.")
        continue
    
    # Break the loop if the user types 'exit'
    if sample_path.lower() == 'exit':
        break

    # Preprocess the input video
    audio_features = pp.get_audio_features(sample_path)
    audio_features = np.expand_dims(audio_features, axis=0)
    audio_features = np.transpose(audio_features, (0, 2, 1))

    print(audio_features.shape) 
    
    # Predict
    predictions = model.predict(audio_features)
    
    # Convert the probability to percentage
    percentage = predictions[0][1] * 100  # Adjust indexing based on model's output shape
    print(f"Prediction: {percentage:.2f}% Liar")



