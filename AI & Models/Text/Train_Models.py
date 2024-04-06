import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import Preprocess as pp
import Feature_Extraction as fe
import Models as md
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn import naive_bayes
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


## Read data
df = pp.get_df_court_trial()

# Shuffle the DataFrame rows
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

dataset_name = "Court_trial"

### Court trial manual split ###
# Define the path to  'manual split test videos.txt' file
test_videos_file_path = '../Datasets/Real Life Trial Cases Data/Manual Split Test Videos.txt'

# Read the test video names from the file
test_video_names = pp.read_test_video_names(test_videos_file_path)

# Split the DataFrame into train and test based on these names
train_df, test_df = pp.split_df_based_on_test_names(df, test_video_names)


### For MU3D Data ###
# dataset_name = "MU3D"
# df_miami = pp.get_df_mu3d()
# train_df, test_df = train_test_split(df_miami, test_size = 0.2, random_state=542, stratify= df_miami['label'])


# Preprocess each df (and save to a file)
train_df = pp.preprocess_df(train_df, "train", num_grams=1)
test_df = pp.preprocess_df(test_df, "test", num_grams= 1)


# Extract features
feature_extractor = 'tf-idf'

# Vectorize using tf idf or count vectorizer or glove
X_train_extracted = fe.tf_idf_vectorize(train_df['text'])
X_test_extracted = fe.tf_idf_vectorize(test_df['text'], is_test= True)

# X_train_extracted = fe.glove_vectorize(train_df['text'])
# X_test_extracted = fe.glove_vectorize(test_df['text'])

# # Save extracted data
# np.save(f'Glove/Extracted data/{dataset_name}glove_100d_x_train.npy', X_train_extracted)
# np.save(f'Glove/Extracted data/{dataset_name}glove_100d_x_test.npy', X_test_extracted)

# X_train_extracted = np.load(f'Glove/Extracted data/{dataset_name}glove_100d_x_train.npy')
# X_test_extracted = np.load(f'Glove/Extracted data/{dataset_name}glove_100d_x_test.npy')


### Reshaping section (for RNNs only) ###

# # Save original shape for model first layer input shape specification.
# model_shape = X_train_extracted.shape

# # Convert to arrays and reshape for the sequential models.
X_train_extracted= X_train_extracted.toarray()
# X_train_extracted = np.reshape(X_train_extracted, newshape=(X_train_extracted.shape[0],1, X_train_extracted.shape[1]))
X_test_extracted= X_test_extracted.toarray()
# X_test_extracted = np.reshape(X_test_extracted, newshape=(X_test_extracted.shape[0],1, X_test_extracted.shape[1]))


# # Print all shapes
# print("Neural Network shape:", model_shape)
# print("Train set shape:", X_train_extracted.shape)
# print("Validation set shape:", X_test_extracted.shape)



# # Target value of each set
y_train = train_df['label']
y_test = test_df['label']


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
md.train_model_dl(ann_model ,f'Saved Models/{dataset_name}' , X_train_extracted, y_train, X_test_extracted, y_test, name =f"ANN {feature_extractor}", batch_size=2, epochs= 10)



## MACHINE LEARNING MODELS ##

#1 SVM
svm_classifier = SVC(kernel='linear', C = 10)
md.train_model_ml(svm_classifier, os.path.join("Saved Models", dataset_name) ,X_train_extracted, y_train, X_test_extracted, y_test, name = f"SVM {feature_extractor}")

#2 Random Forest
rf_classifier = RandomForestClassifier()
md.train_model_ml(rf_classifier, os.path.join("Saved Models", dataset_name) ,X_train_extracted, y_train, X_test_extracted, y_test, name = f"Random Forest {feature_extractor}")

#3 Logistic Regression
lr_classifier = LogisticRegression(max_iter = 3000)
md.train_model_ml(lr_classifier, os.path.join("Saved Models", dataset_name) ,X_train_extracted, y_train, X_test_extracted, y_test, name = f"Logistic Regression {feature_extractor}")

#4 Decision Tree
dt_classifier = DecisionTreeClassifier()
md.train_model_ml(dt_classifier, os.path.join("Saved Models", dataset_name) ,X_train_extracted, y_train, X_test_extracted, y_test, name = f"Decision Tree {feature_extractor}")

#5 GBC
gb_classifier = GradientBoostingClassifier()
md.train_model_ml(gb_classifier, os.path.join("Saved Models", dataset_name) ,X_train_extracted, y_train, X_test_extracted, y_test, name = f"GBC {feature_extractor}")

#6 SGD
sgd_classifier = SGDClassifier()
md.train_model_ml(sgd_classifier, os.path.join("Saved Models", dataset_name) ,X_train_extracted, y_train, X_test_extracted, y_test, name = f"SGD {feature_extractor}")

#7 XGBoost 
xgb_classifier = XGBClassifier()
md.train_model_ml(xgb_classifier, os.path.join("Saved Models", dataset_name) ,X_train_extracted, y_train, X_test_extracted, y_test, name = f"XGB {feature_extractor}")
