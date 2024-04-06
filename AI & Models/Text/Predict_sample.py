import Preprocess as pp
from keras.models import load_model
import numpy as np
import os
import tensorflow as tf
import moviepy.editor as mp
import Feature_Extraction as fe
import speech_recognition as sr

# Path to the saved model
base_path = 'Saved Models/Court_trial'
model_path = os.path.join(base_path, 'ANN tf-idf 0.50 Lss 79.17 Acc.h5') # best generalization

# Load the saved model
model = load_model(model_path)

# Initialize preprocessor class
en_pre = pp.EnglishPreprocessor()


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

    # Load the video 
    video = mp.VideoFileClip(sample_path) 
    
    # Extract the audio from the video 
    audio_file = video.audio 
    audio_file_path = f"{sample_path[:-4]}.wav"
    audio_file.write_audiofile(audio_file_path) 
    
    # Initialize recognizer 
    recognizer = sr.Recognizer() 
    
    # Load the audio file 
    with sr.AudioFile(audio_file_path) as source: 
        audio_data = recognizer.record(source) 
    
    # Convert audio to text
    try:
        text = recognizer.recognize_google(audio_data)
        print("Text extracted from video:", text)
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        
        continue
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        continue
    finally:
        # Remove the temp file 
        os.remove(audio_file_path)

    # Print the text 
    print("\nThe resulting text from video is: \n") 
    print(text) 
    
    # Preprocess the text and extract features
    text = en_pre.preprocess(text)
    text = fe.tf_idf_vectorize([text], is_test= True)
    text = text.toarray()
    
    
    # Predict
    predictions = model.predict(text)
    
    # Convert the probability to percentage
    percentage = predictions[0][0] * 100  # Adjust indexing based on model's output shape
    print(f"Prediction: {percentage:.2f}% Liar")
    
   




