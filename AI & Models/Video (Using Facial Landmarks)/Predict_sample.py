import Preprocess as pp
from keras.models import load_model
import numpy as np

# Path to the saved model
base_path = 'Saved Processed Data and Models/court_trial 100x64x64x1'
model_path = base_path + '/CNN_LSTM 85.71 Acc [BEST].h5' # best generalization

# Load the saved model
model = load_model(model_path)

while True:
    print("Enter sample path (or type 'exit' to quit):")
    sample_path = input()

    # Break the loop if the user types 'exit'
    if sample_path.lower() == 'exit':
        break

    # Preprocess the input video
    processed_frames = pp.preprocess_video(sample_path)
    processed_frames = np.expand_dims(processed_frames, axis=0)
    print(processed_frames.shape) 
    
    # Predict
    predictions = model.predict(processed_frames)
    
    # Convert the probability to percentage
    percentage = predictions[0][0] * 100  # Adjust indexing based on your model's output shape
    print(f"Prediction: {percentage:.2f}% Lier")
