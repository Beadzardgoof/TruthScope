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
df = pp.get_df_mu3d()
dataset_name = "MU3D"

# Split into training and (validation + test)
train_df, val_df = train_test_split(df, test_size = 0.07, random_state=542)

# Preprocess each df (and save to a file)
train_df = pp.preprocess_df(train_df, "train")
val_df = pp.preprocess_df(val_df, "val")


# Extract features

# Vectorize using tf idf or count vectorizer
X_train_tf_idf = fe.count_vectorize(train_df['text'])
X_val_tf_idf = fe.count_vectorize(val_df['text'], is_test= True)


### Reshaping section (for RNNs only) ###

# # Save original shape for model first layer input shape specification.
# model_shape = X_train_tf_idf.shape

# # Convert to arrays and reshape for the sequential models.
# X_train_tf_idf= X_train_tf_idf.toarray()
# X_train_tf_idf = np.reshape(X_train_tf_idf, newshape=(X_train_tf_idf.shape[0],1, X_train_tf_idf.shape[1]))
# X_val_tf_idf= X_val_tf_idf.toarray()
# X_val_tf_idf = np.reshape(X_val_tf_idf, newshape=(X_val_tf_idf.shape[0],1, X_val_tf_idf.shape[1]))


# # Print all shapes
# print("Neural Network shape:", model_shape)
# print("Train set shape:", X_train_tf_idf.shape)
# print("Validation set shape:", X_val_tf_idf.shape)



# Target value of each set
y_train = train_df['label']
y_val = val_df['label']


# Train and evaluate models

## DEEP LEARNING MODELS ##

#1 LSTM
# print("#1 LSTM model \n")
# lstm_model = md.build_lstm(input_shape=(1, model_shape[1]))
# md.train_model_dl(lstm_model,f'Saved Models/{dataset_name}' , X_train_tf_idf, y_train, X_val_tf_idf, y_val, name ="LSTM", batch_size=4, epochs= 20)
# print("\n")

# #2 Bidirectional LSTM
# print("#2 Bidirectional LSTM model \n")
# bidirectional_lstm_model = md.build_lstm_bidirectional(input_shape=(1, model_shape[1]))
# md.train_model(bidirectional_lstm_model,f'Saved Models/{dataset_name}' , X_train_tf_idf, y_train, X_val_tf_idf, y_val, name ="Bidirectional LSTM", batch_size=4, epochs= 40)
# print("\n")

# #3 Simple RNN
# print("#3 Simple RNN model \n")
# simple_rnn_model = md.build_simple_rnn(input_shape=(1, model_shape[1]))
# md.train_model(simple_rnn_model,f'Saved Models/{dataset_name}' , X_train_tf_idf, y_train, X_val_tf_idf, y_val, name ="Simple RNN", batch_size=4, epochs= 40)
# print("\n")

# #4 GRU
# print("#4 GRU model \n")
# gru_model = md.build_gru(input_shape=(1, model_shape[1]))
# md.train_model(gru_model,f'Saved Models/{dataset_name}' , X_train_tf_idf, y_train, X_val_tf_idf, y_val, name ="GRU", batch_size=4, epochs= 40)




## MACHINE LEARNING MODELS ##

#1 SVM
svm_classifier = SVC(kernel='linear', C = 10)
md.train_model_ml(svm_classifier, os.path.join("Saved Models", dataset_name) ,X_train_tf_idf, y_train, X_val_tf_idf, y_val, name = "SVM")

#2 Random Forest
rf_classifier = RandomForestClassifier(n_estimators=50, max_leaf_nodes=10,max_depth=1000)
md.train_model_ml(rf_classifier, os.path.join("Saved Models", dataset_name) ,X_train_tf_idf, y_train, X_val_tf_idf, y_val, name = "Random Forest")

#3 Logistic Regression
lr_classifier = LogisticRegression(max_iter = 1000)
md.train_model_ml(lr_classifier, os.path.join("Saved Models", dataset_name) ,X_train_tf_idf, y_train, X_val_tf_idf, y_val, name = "Logistic Regression")

#4 Decision Tree
dt_classifier = DecisionTreeClassifier(max_depth=5, min_samples_split=2, min_samples_leaf=1)
md.train_model_ml(dt_classifier, os.path.join("Saved Models", dataset_name) ,X_train_tf_idf, y_train, X_val_tf_idf, y_val, name = "Decision Tree")

#5 GBC
gb_classifier = GradientBoostingClassifier(n_estimators=50, learning_rate= 0.1, max_depth=5, random_state=42)
md.train_model_ml(gb_classifier, os.path.join("Saved Models", dataset_name) ,X_train_tf_idf, y_train, X_val_tf_idf, y_val, name = "GBC")

#6 SGD
sgd_classifier = SGDClassifier()
md.train_model_ml(sgd_classifier, os.path.join("Saved Models", dataset_name) ,X_train_tf_idf, y_train, X_val_tf_idf, y_val, name = "SGD")

#7 XGBoost 
xgb_classifier = XGBClassifier()
md.train_model_ml(xgb_classifier, os.path.join("Saved Models", dataset_name) ,X_train_tf_idf, y_train, X_val_tf_idf, y_val, name = "XGB")
