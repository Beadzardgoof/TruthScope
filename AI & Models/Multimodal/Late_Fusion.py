from keras.models import load_model
import Preprocess_Audio as ppa
import Preprocess_Text as ppt
import Preprocess_Video as ppv
import Feature_Extraction_Text as fet 
import numpy as np
import tensorflow as tf
from scipy.stats import mode
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Load the pretrained models on different modalities and print architectures

## Visual (Image/Video based) model
video_model = load_model('Pretrained Models\Video CNN_3D 0.61 Lss 87.50 Acc.h5')
video_model.summary()


## Non visual (Speech/Audio and Text based) models
audio_model = load_model('Pretrained Models/Audio Bidirectional LSTM 0.62 Lss 79.17 Acc.h5')
audio_model.summary()

text_model = load_model('Pretrained Models/Text ANN tf-idf 0.50 Lss 79.17 Acc.h5')
text_model.summary()



# Get preprocessed data in different modalities

### Note, data is not shuffeld and it is read in the same order in all modalities to ensure correct results ###

## Video

# Get data (Manually split)
data_path = "../video/Saved Processed Data and Models/court_trial 100x64x64x1 MTCNN/Numpy Arrays"
manual_split_video_names_path = '../Datasets/Real Life Trial Cases Data/Manual Split Test Videos.txt'

X_train_video, X_test_video, y_train_video, y_test_video = ppv.get_manual_split_data(data_path, manual_split_video_names_path)


## Audio

# Get data (Manually split)
data_path = "../audio/Saved Processed Data and Models/court_trial (frame_length=0.025 hop_length=0.01 num_samples= 700)/Numpy Arrays"

X_train_audio, X_test_audio, y_train_audio, y_test_audio = ppa.get_manual_split_data(data_path, manual_split_video_names_path)

# Reshaping
X_train_audio = np.transpose(X_train_audio, (0, 2, 1))
X_test_audio = np.transpose(X_test_audio, (0, 2, 1))

## Text

## Read data
df = ppt.get_df_court_trial()

# Define the names of the videos to be removed (Because they are removed for video/audio models as they were corrupt)
videos_to_remove = ['trial_lie_045', 'trial_lie_050', 'trial_lie_053']

# Remove the rows where the 'name' column matches any of the values in videos_to_remove
df = df[~df['name'].isin(videos_to_remove)]

### Court trial manual split ###

# Read the test video names from the file
test_video_names = ppt.read_test_video_names(manual_split_video_names_path)

# Split the DataFrame into train and test based on these names
train_df, test_df = ppt.split_df_based_on_test_names(df, test_video_names)

# Preprocess each df (and save to a file)
train_df = ppt.preprocess_df(train_df, "train", num_grams=1)
test_df = ppt.preprocess_df(test_df, "test", num_grams= 1)

# Vectorize using tf idf (Saved already)
X_train_text_extracted = fet.tf_idf_vectorize(train_df['text'], is_test = True)
X_test_text_extracted = fet.tf_idf_vectorize(test_df['text'], is_test= True)

# Convert to arrays
X_train_text_extracted= X_train_text_extracted.toarray()
X_test_text_extracted= X_test_text_extracted.toarray()

# Target value of each set
y_train_text = train_df['label']
y_test_text = test_df['label']




# Getting predictions

# Get predictions for the video data
with tf.device('/cpu:0'): # For using CPU instead of GPU (CPU has more memory but GPU is faster)
    predictions_train_video = video_model.predict(X_train_video).ravel()
    predictions_test_video = video_model.predict(X_test_video).ravel()

# Get predictions for the audio data (Taking second num because it is softmax not sigmoid)
predictions_train_audio = audio_model.predict(X_train_audio)[:,1].ravel()
predictions_test_audio = audio_model.predict(X_test_audio)[:,1].ravel()

# Get predictions for the text data
predictions_train_text = text_model.predict(X_train_text_extracted).ravel()
predictions_test_text = text_model.predict(X_test_text_extracted).ravel()



# Apply threshold to convert probabilities to binary class predictions
threshold = 0.5
predictions_train_video_thresholded = (predictions_train_video > threshold).astype(int)
predictions_test_video_thresholded = (predictions_test_video > threshold).astype(int)
predictions_train_audio_thresholded = (predictions_train_audio > threshold).astype(int)
predictions_test_audio_thresholded = (predictions_test_audio > threshold).astype(int)
predictions_train_text_thresholded = (predictions_train_text > threshold).astype(int)
predictions_test_text_thresholded = (predictions_test_text > threshold).astype(int)



### 1- Majority voting ###

# Majority voting on train data
majority_vote_train = mode(np.c_[predictions_train_video_thresholded, predictions_train_audio_thresholded, predictions_train_text_thresholded], axis=1)[0].flatten()

# Majority voting on test data
majority_vote_test = mode(np.c_[predictions_test_video_thresholded, predictions_test_audio_thresholded, predictions_test_text_thresholded], axis=1)[0].flatten()


### 2- Weighted voting ###

# Define weights for each modality based on validation performance
#weights = [0.15, 0.5, 0.35]  # Weights for video, audio, text (best weights so far)
weights = [0.4, 0.3, 0.3] 

# Weighted voting on train data
weighted_vote_train = np.average(np.c_[predictions_train_video, predictions_train_audio, predictions_train_text], axis=1, weights=weights)
weighted_vote_train = (weighted_vote_train > threshold).astype(int)

# Weighted voting on test data
weighted_vote_test = np.average(np.c_[predictions_test_video, predictions_test_audio, predictions_test_text], axis=1, weights=weights)
weighted_vote_test = (weighted_vote_test > threshold).astype(int)


### 3- Concatenation fusion ###

# Concatenate predictions along axis=1 (side by side)
X_train_concat = np.c_[predictions_train_video, 
                       predictions_train_audio, 
                       predictions_train_text_thresholded]

X_test_concat = np.c_[predictions_test_video, 
                      predictions_test_audio, 
                      predictions_test_text]

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Initialize and train the SVM classifier
svm_classifier = SVC(kernel='rbf', C = 1)
svm_classifier.fit(X_train_concat, y_train_video)

# Make final predictions
final_predictions_train = svm_classifier.predict(X_train_concat)
final_predictions_test = svm_classifier.predict(X_test_concat)






# Evaluate Majority Voting
accuracy_majority_train = accuracy_score(y_train_video, majority_vote_train)  # Assuming same labels for all modalities
accuracy_majority_test = accuracy_score(y_test_video, majority_vote_test)

print(f"Majority Voting Train Accuracy: {accuracy_majority_train}")
print(f"Majority Voting Test Accuracy: {accuracy_majority_test}")

# Evaluate Weighted Voting
accuracy_weighted_train = accuracy_score(y_train_video, weighted_vote_train)
accuracy_weighted_test = accuracy_score(y_test_video, weighted_vote_test)

print(f"Weighted Voting Train Accuracy: {accuracy_weighted_train}")
print(f"Weighted Voting Test Accuracy: {accuracy_weighted_test}")



# Evaluate the model
train_accuracy = accuracy_score(y_train_video, final_predictions_train)
test_accuracy = accuracy_score(y_test_video, final_predictions_test)

print(f"Concatenation Fusion SVM Train Accuracy: {train_accuracy}")
print(f"Concatenation Fusion SVM Test Accuracy: {test_accuracy}")
