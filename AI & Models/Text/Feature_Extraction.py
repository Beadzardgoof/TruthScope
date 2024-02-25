import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from joblib import dump, load

def tf_idf_vectorize(x, is_test = False):
    if (is_test):
        vectorizer = load('TF_IDF_vectorizer.joblib')
        x = vectorizer.transform(x)
    else:
        vectorizer = TfidfVectorizer()
        x = vectorizer.fit_transform(x)
        dump(vectorizer,  'TF_IDF_vectorizer.joblib')
    return x


def count_vectorize(x, is_test=False):
    if is_test:
        vectorizer = load('Count_vectorizer.joblib')
        x = vectorizer.transform(x)
    else:
        vectorizer = CountVectorizer()
        x = vectorizer.fit_transform(x)
        dump(vectorizer, 'Count_vectorizer.joblib')
    return x

