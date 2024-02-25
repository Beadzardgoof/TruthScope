import numpy as np
import pandas as pd
from nltk.tokenize import ToktokTokenizer
import string
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, SnowballStemmer
import re
import nltk
import re 
import os
#nltk.download('wordnet')
#nltk.download('stopwords')


# Helper functions to prepare text and labels dataframes
def get_df_court_trial():
    base_path = "../Datasets/Real Life Trial Cases Data/Transcription"
    
    categories = ["Deceptive", "Truthful"]
    texts_labels = []  # List to store text content and labels

    for category in categories:
        category_path = os.path.join(base_path, category)
        for text_file in os.listdir(category_path):
            text_path = os.path.join(category_path, text_file)
            label = 1 if category == "Deceptive" else 0
            # Read the content of text file
            with open(text_path, 'r', encoding='utf-8') as file:
                text = file.read()
                texts_labels.append((text, label))

    # Create a DataFrame with the text and labels
    df = pd.DataFrame(texts_labels, columns=['text', 'label'])
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




# NLP pipeline class 
class EnglishPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.tokenizer=ToktokTokenizer()
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

    def preprocess(self, text):
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Tokenize text and transform to lower cass
        tokens = word_tokenize(text.lower())

        # Remove stop words
        filtered_tokens = [token for token in tokens if token not in self.stop_words]

        # Lemmatize/stem
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in filtered_tokens]

        # Rejoin tokens into a string
        return ' '.join(lemmatized_tokens)



# Main function (Optionally define number of samples to preprocess, to process smaller batches for testing if required)
def preprocess_df(df, out_name, isPredict = False):
    # Preprocess text data
    en_pre = EnglishPreprocessor()
    
    # List for processed text
    processed_texts = []
    
    # Iterate over text and preprocess according to language
    for text in df['text']:
        processed_text = en_pre.preprocess(text)
        processed_texts.append(processed_text)
    
    # Replace text with preprocessed version
    df['text'] = processed_texts
    
    # Preprocess target values and save preprocessed data (only if data is labeled, i.e in training and testing, not prediction)
    if not isPredict:
        # Save to Excel
        df.to_excel(f'preprocessed_{out_name}.xlsx', index=False)
    
    return df


