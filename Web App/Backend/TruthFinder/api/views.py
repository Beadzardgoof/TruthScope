from django.http import JsonResponse
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.decorators import api_view, parser_classes
from rest_framework import status
import numpy as np
from api.Preprocess_Video import *
from api.Preprocess_Text import *
from api.Preprocess_Audio import *
from api.Preprocess_Expressions import *
from keras.models import load_model
import os
import tensorflow as tf
import moviepy.editor as mp
from api.Feature_Extraction_Text import *
import speech_recognition as sr
import statistics

@api_view(['POST'])
@parser_classes((MultiPartParser, FormParser))
def upload_video(request):
    if 'file' not in request.data:
        return JsonResponse({'message': 'No video file provided'}, status=status.HTTP_400_BAD_REQUEST)
    
    video_file = request.data['file']
    file_path = f"uploaded videos/{video_file.name}"
    os.makedirs('uploaded videos', exist_ok=True)  # This will create the directory if it does not exist, without raising an error if it already exists.
    with open(file_path, 'wb+') as destination:
        for chunk in video_file.chunks():
            destination.write(chunk)

    # Processing the video and getting the prediction result
    result = process_video(file_path) 
   
    return JsonResponse({'message': 'Video processed successfully', 'result': result})


# Takes path to a video file and applies inference on all modalities in addition to late fusion
def process_video(sample_path):
    
    # Set the environment variable for reproducability
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

    ## load the models
    video_model = load_model('api/Pretrained Models/Video CNN_3D 0.61 Lss 87.50 Acc.h5')
    expressions_model = load('api/Pretrained Models/Expressions XGB 75.00 Acc.joblib')
    audio_model = load_model('api/Pretrained Models/Audio Bidirectional LSTM 0.62 Lss 79.17 Acc.h5')
    text_model = load_model('api/Pretrained Models/Text ANN tf-idf 0.50 Lss 79.17 Acc.h5')


    # Initialize the text preprocessor class
    en_pre = EnglishPreprocessor()

    ### Video ###
    
    # Preprocess the input video
    processed_frames = preprocess_video_mtcnn(sample_path)
    save_frames_as_gif(processed_frames, 'visualize frames')
    processed_frames = np.expand_dims(processed_frames, axis=0)
    
    # Predict
    predictions_video = video_model.predict(processed_frames)
    
    # Convert the probability to percentage
    percentage_video = predictions_video[0][0] * 100  # Adjust indexing based on model's output shape
    print(f"Prediction (Video): {percentage_video:.2f}% Liar")

    ### Expressions ###
    
    # Extract features
    output_path = get_video_name_from_path(sample_path) + "_features.csv"
    fex = extract_facial_features(sample_path, output_path)
        
    # Prepare feature vector
    feature_vector = prepare_feature_vector(output_path)
    
    feature_vector = np.expand_dims(feature_vector, axis=0)
    os.remove(output_path)
        
    # Predict
    predictions_expressions = expressions_model.predict_proba(feature_vector)
    
    # Convert the probability to percentage
    percentage_expressions = predictions_expressions[0][0] * 100  # Adjust indexing based on model's output shape
    print(f"Prediction (expressions): {percentage_expressions:.2f}% Liar")

    ### AUDIO ###
    
    # Preprocess the input video
    audio_features = get_audio_features(sample_path)
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
        text = tf_idf_vectorize([text], is_test= True)
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
    
    # Threshold all votes
    prediction_video_thresholded = (percentage_video > threshold).astype(int)
    predictions_expressions_thresholded = (percentage_expressions > threshold).astype(int)
    prediction_audio_thresholded = (percentage_audio > threshold).astype(int)
        
    
    # Perform both majority and weigthed average voting with special cases if transcription fails/succeeds
    if isTranscribed:
        # Threshold text
        prediction_text_thresholded = (percentage_text > threshold).astype(int)

        ## Majority voting (video votes twice to get odd number of votes)
        majority_vote = statistics.mode([prediction_video_thresholded, prediction_video_thresholded, predictions_expressions_thresholded, prediction_audio_thresholded, prediction_text_thresholded])
        
        ## Weighted average
        weights = [0.4, 0.2, 0.2, 0.2] # Weights for video, expressions, audio and text
        weighted_average = weights[0] * percentage_video + weights[1] * percentage_expressions + weights[2] * percentage_audio + weights[3] * percentage_text
        
    else:
        ## Majority voting
        majority_vote = statistics.mode([prediction_video_thresholded, predictions_expressions_thresholded, prediction_audio_thresholded])
        
        ## Weighted average
        weights = [0.4, 0.3,0.3]  # Weights for video, expressions and audio
        weighted_average = weights[0] * percentage_video + weights[1] * percentage_expressions + weights[2] * percentage_audio
        
    
    # Print majority voting result
    if majority_vote == 1:
        print("Prediction(Majority Voting Late Fusion): Liar")
    else:
        print("Prediction(Majority Voting Late Fusion): Not a Liar")
            
    
    # Print weighted average result
    print(f"Prediction(Weighted average Late Fusion): {weighted_average:.2f}% Liar")
    
    
    # Creating the dictionary to be serialized to JSON for the HTTP response
    # Converting NumPy types to native Python types for JSON serialization
    data = {
    "percentage_video": round(float(percentage_video), 2),
    "percentage_expressions": round(float(percentage_expressions), 2),
    "percentage_audio": round(float(percentage_audio), 2),  
    "percentage_text": round(float(percentage_text), 2) if isTranscribed else None, 
    "majority_vote": int(majority_vote),  
    "weighted_average": round(float(weighted_average), 2)  
}
    
    return data