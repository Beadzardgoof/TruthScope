import numpy as np
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, Conv3D, MaxPooling3D, Flatten, LSTM, Bidirectional, Multiply
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import RMSprop
import Preprocess_Audio as ppa
import Preprocess_Text as ppt
import Preprocess_Video as ppv
import Feature_Extraction_Text as fet
import os

# Load and preprocess the data for each modality

## Video
data_path = "../video/Saved Processed Data and Models/court_trial 100x64x64x1 MTCNN/Numpy Arrays"
manual_split_video_names_path = '../Datasets/Real Life Trial Cases Data/Manual Split Test Videos.txt'
X_train_video, X_test_video, y_train_video, y_test_video = ppv.get_manual_split_data(data_path, manual_split_video_names_path)

## Audio
data_path = "../audio/Saved Processed Data and Models/court_trial (frame_length=0.025 hop_length=0.01 num_samples= 700)/Numpy Arrays"
X_train_audio, X_test_audio, y_train_audio, y_test_audio = ppa.get_manual_split_data(data_path, manual_split_video_names_path)
X_train_audio = np.transpose(X_train_audio, (0, 2, 1))
X_test_audio = np.transpose(X_test_audio, (0, 2, 1))

## Text
df = ppt.get_df_court_trial()
videos_to_remove = ['trial_lie_045', 'trial_lie_050', 'trial_lie_053']
df = df[~df['name'].isin(videos_to_remove)]
test_video_names = ppt.read_test_video_names(manual_split_video_names_path)
train_df, test_df = ppt.split_df_based_on_test_names(df, test_video_names)
train_df = ppt.preprocess_df(train_df, "train", num_grams=1)
test_df = ppt.preprocess_df(test_df, "test", num_grams=1)
X_train_text_extracted = fet.bert_vectorize(train_df['text'].tolist())
X_test_text_extracted = fet.bert_vectorize(test_df['text'].tolist())
#X_train_text_extracted = fet.tf_idf_vectorize(train_df['text'], is_test=True).toarray()
#X_test_text_extracted = fet.tf_idf_vectorize(test_df['text'], is_test=True).toarray()
y_train_text = train_df['label']
y_test_text = test_df['label']

# Define the model
input_video = Input(shape=(100, 64, 64, 1), name='video_input')
conv3d_1 = Conv3D(32, kernel_size=(1, 3, 3), activation='relu')(input_video)
maxpool3d_1 = MaxPooling3D(pool_size=(2, 2, 2))(conv3d_1)
conv3d_2 = Conv3D(32, kernel_size=(3, 1, 1), activation='relu')(maxpool3d_1)
maxpool3d_2 = MaxPooling3D(pool_size=(2, 2, 2))(conv3d_2)
flatten_video = Flatten()(maxpool3d_2)
dense_video = Dense(500, activation='relu')(flatten_video)
dropout_video = Dropout(0.2)(dense_video)
dense_video_2 = Dense(100, activation='relu')(dropout_video)


input_audio = Input(shape=(X_train_audio.shape[1], X_train_audio.shape[2]), name='audio_input')
lstm_audio_1 = Bidirectional(LSTM(32, activation='tanh'))(input_audio)
dense_audio = Dense(100, activation='relu')(lstm_audio_1)

input_text = Input(shape=(X_train_text_extracted.shape[1],), name='text_input')
dense_text = Dense(100, activation='relu')(input_text)

# Use Hadamard product for concatenation
hadamard_product = Multiply()([dense_video_2, dense_audio, dense_text])

dense_1 = Dense(300, activation='relu')(hadamard_product)
dropout_1 = Dropout(0.2)(dense_1)

dense_2 = Dense(100, activation='relu')(dropout_1)
dropout_2 = Dropout(0.2)(dense_2)

output = Dense(1, activation='sigmoid')(dropout_2)

model = Model(inputs=[input_video, input_audio, input_text], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

output_path = "Early Fusion"
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Define checkpoint
checkpoint = ModelCheckpoint(
    os.path.join(output_path, 'Checkpoint.h5'), save_weights_only=False, save_best_only=True, verbose=1, monitor='val_accuracy', mode='max',
)

model.fit([X_train_video, X_train_audio, X_train_text_extracted], y_train_video, epochs=30, batch_size=4, validation_data=([X_test_video, X_test_audio, X_test_text_extracted], y_test_video), callbacks=[checkpoint])

# Evaluate the model
with tf.device('/cpu:0'):
    model_checkpoint = load_model(os.path.join(output_path, 'Checkpoint.h5'))
    evaluation = model_checkpoint.evaluate([X_test_video, X_test_audio, X_test_text_extracted], y_test_video)

# Rename the checkpoint file to include its metrics
old_path = os.path.join(output_path, 'Checkpoint.h5')
new_path = os.path.join(output_path, "MLP" + ' {:.2f} Lss'.format(evaluation[0]) + ' {:.2f} Acc.h5'.format(evaluation[1] * 100))
os.rename(old_path, new_path)
