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
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler


### Court trial dataset ###

## Prepare data

output_path = "Saved Processed Data and Models/court_trial (frame_length=0.025 hop_length=0.01 num_samples= 700)"
#pp.prepare_dataset(output_folder= output_path, dataset= 'court_trial')

# Get data and manually split
data_path = "Saved Processed Data and Models/court_trial (frame_length=0.025 hop_length=0.01 num_samples= 700)/Numpy Arrays"
manual_split_video_names_path = '../Datasets/Real Life Trial Cases Data/Manual Split Test Videos.txt'

# Manual Split
X_train, X_test, y_train, y_test = pp.get_manual_split_data(data_path, manual_split_video_names_path)

# # Auto split (For testing)
# #X, y = pp.get_data_from_saved_numpy_arrays(data_path)
# #X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2 ,stratify=y)

# # PCA
# #X_train = pp.scale_and_apply_pca(X_train, num_pca_components=5, is_test = False)
# #X_test = pp.scale_and_apply_pca(X_test, num_pca_components=5, is_test = True)


### MU3D ###

# # Prepare data
# output_path = "Saved Processed Data and Models/MU3D (frame_length=0.025 hop_length=0.01 num_samples= 700)"
# pp.prepare_dataset(output_folder= output_path, dataset= 'MU3D')


# # Train test split
# data_path = "Saved Processed Data and Models/MU3D (frame_length=0.025 hop_length=0.01 num_samples= 700)/Numpy Arrays"
# X, y = pp.get_data_from_saved_numpy_arrays(data_path)
# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2 ,stratify=y, shuffle= True)


### Reshaping section (for RNNs only) ###

model_input_shape = (X_train.shape[2], X_train.shape[1])
X_train = np.transpose(X_train, (0, 2, 1))
X_test = np.transpose(X_test, (0, 2, 1))

# Printing shapes
print('X_train shape is:' , X_train.shape)
print('X_test shape is:' , X_test.shape)


# Train and evaluate models

## DEEP LEARNING MODELS ##

## Sequential Models ##

# #1 LSTM
# print("#1 LSTM model \n")
# lstm_model = md.build_lstm(input_shape=model_input_shape)
# md.train_model_dl(lstm_model, output_path , X_train, y_train, X_test, y_test, name ="LSTM", batch_size=4, epochs= 40)
# print("\n")

# #2 Bidirectional LSTM

# print("#2 Bidirectional LSTM model \n")
# bidirectional_lstm_model = md.build_lstm_bidirectional(input_shape=model_input_shape)
# md.train_model_dl(bidirectional_lstm_model, output_path, X_train, y_train, X_test, y_test, name ="Bidirectional LSTM", batch_size=4, epochs= 40)

#3 Simple RNN
# print("#3 Simple RNN model \n")
# simple_rnn_model = md.build_simple_rnn(input_shape=model_input_shape)
# md.train_model_dl(simple_rnn_model, output_path , X_train, y_train, X_test, y_test, name ="Simple RNN", batch_size=4, epochs= 20)
# print("\n")

#4 GRU
print("#4 GRU model \n")
gru_model = md.build_gru(input_shape=model_input_shape)
md.train_model_dl(gru_model, output_path , X_train, y_train, X_test, y_test, name ="GRU", batch_size=4, epochs= 40)


# #5 CNN Biridrectional LSTM
# print("#5 CNN-BiLSTM model \n")
# gru_model = md.build_cnn_lstm_bidirectional(input_shape=model_input_shape)
# md.train_model_dl(gru_model, output_path , X_train, y_train, X_test, y_test, name ="CNN-BiLSTM", batch_size=4, epochs= 40)





# # Flattening by getting the mean for non-sequential algorithms
# X_train = np.mean(X_train, axis=1)
# X_test = np.mean(X_test, axis=1)

# 5 Simple ANN
# print("#5 ANN model \n")
# ann_model = md.build_ann()
# md.train_model_dl(ann_model , output_path , X_train, y_train, X_test, y_test, name ="ANN", batch_size=1, epochs= 10)




## MACHINE LEARNING MODELS ##



# #1 SVM
# svm_classifier = SVC(kernel='rbf', C = 10)
# md.train_model_ml(svm_classifier, output_path ,X_train, y_train, X_test, y_test, name = f"SVM")

# #2 Random Forest
# rf_classifier = RandomForestClassifier(n_estimators=1500, max_leaf_nodes=10,max_depth=50)
# md.train_model_ml(rf_classifier, output_path ,X_train, y_train, X_test, y_test, name = f"Random Forest")

# #3 Logistic Regression
# lr_classifier = LogisticRegression(max_iter = 1000)
# md.train_model_ml(lr_classifier, output_path ,X_train, y_train, X_test, y_test, name = f"Logistic Regression")

# #4 Decision Tree
# dt_classifier = DecisionTreeClassifier(max_depth=5, min_samples_split=2, min_samples_leaf=1)
# md.train_model_ml(dt_classifier, output_path ,X_train, y_train, X_test, y_test, name = f"Decision Tree")

# #5 GBC
# gb_classifier = GradientBoostingClassifier(n_estimators=50, learning_rate= 0.1, max_depth=5, random_state=42)
# md.train_model_ml(gb_classifier, output_path ,X_train, y_train, X_test, y_test, name = f"GBC")

# #6 SGD
# sgd_classifier = SGDClassifier()
# md.train_model_ml(sgd_classifier, output_path ,X_train, y_train, X_test, y_test, name = f"SGD")

# #7 XGBoost 
# xgb_classifier = XGBClassifier()
# md.train_model_ml(xgb_classifier, output_path ,X_train, y_train, X_test, y_test, name = f"XGB")
