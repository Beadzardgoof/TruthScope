import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from joblib import dump, load
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from transformers import BertTokenizer, BertModel
import torch

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

# Converts text documents into mean GloVe embeddings
def glove_vectorize(x):
    
    model = KeyedVectors.load_word2vec_format('Glove/glove.6B.100d.txt.word2vec', binary=False)
    vectors = []
    for doc in x:
        words = doc.split()
        word_vectors = [model[word] for word in words if word in model]
        if len(word_vectors) > 0:
            vectors.append(np.mean(word_vectors, axis=0))
        else:
            # Handling the case where none of the words in the document are in the GloVe model's vocabulary
            vectors.append(np.zeros(model.vector_size))
    
    return np.array(vectors)


def bert_vectorize(texts):
    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Tokenize input text
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    
    # Get hidden states from BERT
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the output embeddings (last hidden state)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Averaging over the sequence length
    return embeddings.numpy()

### To transform a glove.txt file to word2vec format

# # Path to the original GloVe file
# glove_input_file = 'glove/glove.6B.50d.txt'
# # Path to the word2vec output file
# word2vec_output_file = 'glove/glove.6B.50d.txt.word2vec'

# # Convert the GloVe file to word2vec format
# glove2word2vec(glove_input_file, word2vec_output_file)

# # Load the converted file as a Gensim model
# model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)