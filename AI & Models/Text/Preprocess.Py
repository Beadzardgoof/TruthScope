import numpy as np
import pandas as pd
from nltk.tokenize import ToktokTokenizer
import string
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.util import ngrams
import re
import nltk
import re 
import os
nltk.download('wordnet')
nltk.download('stopwords')


# Helper functions to prepare text and labels dataframes
def get_df_court_trial():
    base_path = "../Datasets/Real Life Trial Cases Data/Transcription"
    
    categories = ["Deceptive", "Truthful"]
    df_content = []  # List to store text content and labels and video names

    for category in categories:
        category_path = os.path.join(base_path, category)
        for text_file in os.listdir(category_path):
            text_path = os.path.join(category_path, text_file)
            video_name = os.path.splitext(os.path.basename(text_file))[0]
            label = 1 if category == "Deceptive" else 0
            # Read the content of text file
            with open(text_path, 'r', encoding='utf-8') as file:
                text = file.read()
                df_content.append((video_name,text, label))

    # Create a DataFrame with the text and labels
    df = pd.DataFrame(df_content, columns=['name' ,'text', 'label'])
    return df
    
def get_df_mu3d():
        
    # Load the cookbook
    cookbook_path = "../Datasets/MU3D-Package/MU3D Codebook.xlsx"
    df = pd.read_excel(cookbook_path, sheet_name='Video-Level Data')

    # Get desired columns in a new dataframe
    new_df = pd.DataFrame({
        'text': df['Transcription'],
        'label': df['Veracity']
    })    
    
    return new_df


# Reads a text file containing video names and returns a list of these names.
def read_test_video_names(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        video_names = file.read().splitlines()
    return video_names

# Splits the DataFrame into training and test sets based on the video names.
def split_df_based_on_test_names(df, test_video_names):
    # Filter the DataFrame to create the test set
    test_df = df[df['name'].isin(test_video_names)]
    
    # Filter the DataFrame to create the training set
    train_df = df[~df['name'].isin(test_video_names)]
    
    return train_df, test_df


# NLP pipeline
class EnglishPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.tokenizer=ToktokTokenizer()
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

    def preprocess(self, text, num_grams=1):
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Tokenize text and transform to lower cass
        tokens = word_tokenize(text.lower())
        
        # Remove stop words
        filtered_tokens = [token for token in tokens if token not in self.stop_words]

        # Lemmatize/stem
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in filtered_tokens]

        # Generate n-grams
        if num_grams > 1:
            n_grams = list(ngrams(lemmatized_tokens, num_grams))
            # Rejoin n-grams into a string (each n-gram is joined by a space and then all n-grams are joined into one string)
            return ' '.join([' '.join(gram) for gram in n_grams])
        else:
            # Rejoin tokens into a string for unigrams
            return ' '.join(lemmatized_tokens)



# Main function (Optionally define number of samples to preprocess, to process smaller batches for testing if required)
def preprocess_df(df, out_name, isPredict = False, num_grams = 1):
    # Preprocess text data
    en_pre = EnglishPreprocessor()
    
    # List for processed text
    processed_texts = []
    
    # Iterate over text and preprocess according to language
    for text in df['text']:
        processed_text = en_pre.preprocess(text, num_grams= num_grams)
        processed_texts.append(processed_text)
    
    # Replace text with preprocessed version
    df.loc[:, 'text'] = processed_texts    

    # Save to Excel for visualization
    df.to_excel(f'preprocessed_{out_name}.xlsx', index=False)
    
    return df


