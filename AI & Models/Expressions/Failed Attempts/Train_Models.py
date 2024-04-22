import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import Preprocess as pp
import Models as md
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn import naive_bayes
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle

### Court trial dataset ###
dataset_name = "court_trial"

# Prepare the dataset by extracting signals from all videos
pp.prepare_dataset()

# Extract features from the signals
signals_df = pp.load_data_from_csv("signals_court_trial.csv")
features_df = pp.extract_features(signals_df)
features_df.to_csv("feature_vectors_court_trial.csv", index= False)


### Court trial manual split ###
# Define the path to  'manual split test videos.txt' file
test_videos_file_path = '../Datasets/Real Life Trial Cases Data/Manual Split Test Videos.txt'

# Read the test video names from the file
test_video_names = pp.read_test_video_names(test_videos_file_path)

# Split the DataFrame into train and test based on these names
train_df, test_df = pp.split_df_based_on_test_names(features_df, test_video_names)

# Shuffling dataframes
train_df = shuffle(train_df, random_state= 42)
test_df = shuffle(test_df, random_state= 42)

# Getting feature vectors and labels
X_train = train_df.drop(['Video_ID', 'label'], axis=1)
X_test = test_df.drop(['Video_ID', 'label'], axis=1)
y_train = train_df['label']
y_test = test_df['label']


## Checking correlation + feature selection

# Compute correlation coefficients with label column

correlation = X_train.corrwith(y_train)

# Get the absolute values of correlation coefficients
correlation = correlation.abs()

# Sort the correlation values in descending order
correlation_sorted = correlation.sort_values(ascending=False)
print(correlation_sorted)

# Select top features
top_features = correlation_sorted[:].index


# Printing shapes
print('X_train shape is:' , X_train.shape)
print('X_test shape is:' , X_test.shape)


### Reshaping section (for RNNs only) ###

# # Save original shape for model first layer input shape specification.
# model_shape = X_train_extracted.shape

# # Convert to arrays and reshape for the sequential models.
#X_train_extracted= X_train_extracted.toarray()
# X_train_extracted = np.reshape(X_train_extracted, newshape=(X_train_extracted.shape[0],1, X_train_extracted.shape[1]))
#X_test_extracted= X_test_extracted.toarray()
# X_test_extracted = np.reshape(X_test_extracted, newshape=(X_test_extracted.shape[0],1, X_test_extracted.shape[1]))


# # Print all shapes
# print("Neural Network shape:", model_shape)
# print("Train set shape:", X_train_extracted.shape)
# print("Validation set shape:", X_test_extracted.shape)



# Train and evaluate models

## DEEP LEARNING MODELS ##

# #1 LSTM
# print("#1 LSTM model \n")
# lstm_model = md.build_lstm(input_shape=(1, model_shape[1]))
# md.train_model_dl(lstm_model,f'Saved Models/{dataset_name}' , X_train_extracted, y_train, X_test_extracted, y_test, name ="LSTM", batch_size=4, epochs= 20)
# print("\n")

# #2 Bidirectional LSTM
# print("#2 Bidirectional LSTM model \n")
# bidirectional_lstm_model = md.build_lstm_bidirectional(input_shape=(1, model_shape[1]))
# md.train_model_dl(bidirectional_lstm_model,f'Saved Models/{dataset_name}' , X_train_extracted, y_train, X_test_extracted, y_test, name ="Bidirectional LSTM", batch_size=4, epochs= 20)
# print("\n")

# #3 Simple RNN
# print("#3 Simple RNN model \n")
# simple_rnn_model = md.build_simple_rnn(input_shape=(1, model_shape[1]))
# md.train_model_dl(simple_rnn_model,f'Saved Models/{dataset_name}' , X_train_extracted, y_train, X_test_extracted, y_test, name ="Simple RNN", batch_size=4, epochs= 20)
# print("\n")

# #4 GRU
# print("#4 GRU model \n")
# gru_model = md.build_gru(input_shape=(1, model_shape[1]))
# md.train_model_dl(gru_model,f'Saved Models/{dataset_name}' , X_train_extracted, y_train, X_test_extracted, y_test, name ="GRU", batch_size=4, epochs= 20)


#5 Simple ANN
print("#5 ANN model \n")
ann_model = md.build_ann()
md.train_model_dl(ann_model ,f'Saved Models/{dataset_name}' , X_train, y_train, X_test, y_test, name =f"ANN", batch_size=4, epochs= 20)



## MACHINE LEARNING MODELS ##

#1 SVM
svm_classifier = SVC(kernel='rbf', C = 10)
md.train_model_ml(svm_classifier, os.path.join("Saved Models", dataset_name) ,X_train, y_train, X_test, y_test, name = f"SVM")

#2 Random Forest
rf_classifier = RandomForestClassifier(max_depth=1500,)
md.train_model_ml(rf_classifier, os.path.join("Saved Models", dataset_name) ,X_train, y_train, X_test, y_test, name = f"Random Forest")

#3 Logistic Regression
lr_classifier = LogisticRegression(max_iter = 3000)
md.train_model_ml(lr_classifier, os.path.join("Saved Models", dataset_name) ,X_train, y_train, X_test, y_test, name = f"Logistic Regression")

#4 Decision Tree
dt_classifier = DecisionTreeClassifier()
md.train_model_ml(dt_classifier, os.path.join("Saved Models", dataset_name) ,X_train, y_train, X_test, y_test, name = f"Decision Tree")

#5 GBC
gb_classifier = GradientBoostingClassifier()
md.train_model_ml(gb_classifier, os.path.join("Saved Models", dataset_name) ,X_train, y_train, X_test, y_test, name = f"GBC")

#6 SGD
sgd_classifier = SGDClassifier()
md.train_model_ml(sgd_classifier, os.path.join("Saved Models", dataset_name) ,X_train, y_train, X_test, y_test, name = f"SGD")

#7 XGBoost 
xgb_classifier = XGBClassifier()
md.train_model_ml(xgb_classifier, os.path.join("Saved Models", dataset_name) ,X_train, y_train, X_test, y_test, name = f"XGB")
