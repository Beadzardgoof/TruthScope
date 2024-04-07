import Preprocess_Text as ppt
import Preprocess_Audio as ppa
import Preprocess_Video as ppv
from keras.models import load_model
import numpy as np
import os
import tensorflow as tf
import moviepy.editor as mp
import Feature_Extraction_Text as fe
import speech_recognition as sr
import statistics

# Set the environment variable for reproducability
os.environ['TF_DETERMINISTIC_OPS'] = '1'

## load the models

video_model = load_model('Pretrained Models/Video CNN_3D 0.46 Lss 83.33 Acc.h5')
audio_model = load_model('Pretrained Models/Audio Bidirectional LSTM 0.62 Lss 79.17 Acc.h5')
text_model = load_model('Pretrained Models/Text ANN tf-idf 0.50 Lss 79.17 Acc.h5')


# Initialize the text preprocessor class
en_pre = ppt.EnglishPreprocessor()


print("For best results, your sample should be 20-30 seconds long.")

#with tf.device('/cpu:0'): # For using CPU instead of GPU (CPU has more memory but GPU is faster)
while True:
    print("Enter sample path (or type 'exit' to quit):")
    sample_path = input()

    if not os.path.exists(sample_path):
        print("Please Make sure the path is correct.")
        continue
    
    # Break the loop if the user types 'exit'
    if sample_path.lower() == 'exit':
        break

    ### Video ###
    
    # Preprocess the input video
    processed_frames = ppv.preprocess_video(sample_path, verify_faces= False)
    ppv.save_frames_as_gif(processed_frames, 'visualize')
    processed_frames = np.expand_dims(processed_frames, axis=0)
    
    # Predict
    predictions_video = video_model.predict(processed_frames)
    
    # Convert the probability to percentage
    percentage_video = predictions_video[0][0] * 100  # Adjust indexing based on model's output shape
    print(f"Prediction (Video): {percentage_video:.2f}% Liar")


    ### AUDIO ###
    
    # Preprocess the input video
    audio_features = ppa.get_audio_features(sample_path)
    audio_features = np.expand_dims(audio_features, axis=0)
    audio_features = np.transpose(audio_features, (0, 2, 1))

    # Predict
    predictions_audio = audio_model.predict(audio_features)
    
    # Convert the probability to percentage
    percentage_audio = predictions_audio[0][1] * 100  # Adjust indexing based on model's output shape
    print(f"Prediction (Audio): {percentage_audio:.2f}% Liar")



    ### TEXT ###
    
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

        # Preprocess the text and extract features
        text = en_pre.preprocess(text)
        text = fe.tf_idf_vectorize([text], is_test= True)
        text = text.toarray()

        # Predict
        predictions_text = text_model.predict(text)

        # Convert the probability to percentage
        percentage_text = predictions_text[0][0] * 100  # Adjust indexing based on model's output shape
        print(f"Prediction (Text): {percentage_text:.2f}% Liar")
        
        # Flag to keep track if text Transcribtion succeeded or not
        isTranscribed = True

    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        isTranscribed = False
        
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        isTranscribed = False
    finally:
        # Remove the temp file 
        os.remove(audio_file_path)
        
        
    threshold = 50
    
    # Perform both majority and weigthed average voting if text model succeeded, otherwise do weighted average only
    if isTranscribed:
        ## Majority voting
        
        # Threshold all votes
        prediction_video_thresholded = (percentage_video > threshold).astype(int)
        prediction_audio_thresholded = (percentage_audio > threshold).astype(int)
        prediction_text_thresholded = (percentage_text > threshold).astype(int)
        
        # Get majority vote
        majority_vote = statistics.mode([prediction_video_thresholded, prediction_audio_thresholded, prediction_text_thresholded])
        
        # Print result
        if majority_vote == 1:
            print("Prediction(Majority Voting Late Fusion): Liar")
        else:
            print("Prediction(Majority Voting Late Fusion): Not a Liar")
            
        
        ## Weighted average
        
        weights = [0.33, 0.33, 0.33]
        weighted_average = weights[0] * percentage_video + weights[1] * percentage_audio + weights[2] * percentage_text
        
    else:
        weights = [0.7, 0.3]  # Weights for video, audio
        
        # Get weighted vote and threshold it
        weighted_average = weights[0] * percentage_video + weights[1] * percentage_audio
        
        

    # Print weighted average result
    print(f"Prediction(Weighted average Late Fusion): {weighted_average:.2f}% Liar")

    


   
