import Preprocess as pp
import numpy as np
import os
from joblib import load

# Path to the saved model
base_path = 'Saved Models\Court Trial'
model_path = os.path.join(base_path, 'XGB 75.00 Acc.joblib') # best generalization

# Load the saved model
model = load(model_path)


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

    # Extract features
    output_path = pp.get_video_name_from_path(sample_path) + "_features.csv"
    fex = pp.extract_facial_features(sample_path, output_path)
    
    # Prepare feature vector
    feature_vector = pp.prepare_feature_vector(output_path)
    
    feature_vector = np.expand_dims(feature_vector, axis=0)
    #os.remove(output_path)
    
    print(feature_vector.shape)
    
    # Predict
    predictions = model.predict_proba(feature_vector)
    
    # Convert the probability to percentage
    percentage = predictions[0][0] * 100  # Adjust indexing based on model's output shape
    print(f"Prediction: {percentage:.2f}% Liar")
    
   




