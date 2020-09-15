import numpy as np
import scipy.sparse
import json
import zipfile

from gensim import corpora
from gensim.matutils import corpus2csc
from gensim.corpora.dictionary import Dictionary

from os.path import expanduser
import re, time

from itertools import zip_longest
from operator import itemgetter


def tokenize(x): 
    """regex tokenizer"""
    return re.findall(r'\w+', x)

def create_dataset(
    reviews, 
    categories, 
    num_documents, 
    num_labels, 
    restaurants
) -> np.array:
    
    sparse = np.zeros((num_documents, num_labels))
    for i in range(len(reviews)):
        for item in restaurants[i]['category']:
            sparse[i, categories[item]] = 1
    
    return sparse

def text_sequencer(dictionary, text, max_len=200):
    
    processed = []
    # in case the word is not in the dictionary because it was filtered out use this number to represent an out of set id 
    dict_final = len(dictionary.keys())+1
    
    for i, word in enumerate(text):        
        if i > max_len-1:
            break
        if word in dictionary.token2id.keys():
    # remember the ids have an offset of 1 for this because 0 represents a padded value        
            processed.append(dictionary.token2id[word] + 1) 
        else:
            processed.append(dict_final)
    
    return processed

def load_embeddings(
    glove_data='glove.6B.100d.txt', 
    emb_path = '/projects/embeddings/data/',
    fname='glove.6B.100d.txt'
):
    
    """loads glove vectors"""

    embeddings_index={}
    with zipfile.ZipFile(expanduser("~")+ emb_path + fname, 'r') as f:
        with f.open(glove_data, 'r') as z:
            for line in z:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
    
    return embeddings_index

def id_to_glove(dict_yelp: Dictionary) -> np.array:
    
    """creates embeddings_matrix to be loaded for initial weights"""

    embeddings_index = load_embeddings()
    conversion_table = {}

    for word in dict_yelp.values():
        if bytes(word, 'utf-8') in embeddings_index.keys():
            conversion_table[dict_yelp.token2id[word]+1] = embeddings_index[bytes(word, 'utf-8')]
        else:
            conversion_table[dict_yelp.token2id[word]+1] = np.random.normal(0, .32, 100)
            
    embedding_matrix = np.vstack((np.zeros(100), np.vstack(conversion_table.values()), np.random.randn(100)))
    
    return embedding_matrix

def convert_rating(rating):
    if rating in [4,5]:
        return 1
    elif rating in [1,2]:
        return 0
    else:
        return None
    
def get_rating_set(corpus, stars):
    
    mids = set()
    
    def _get_mids():

        for i, rating in enumerate(stars):
            if rating is None:
                mids.add(i)
    
    _get_mids()
    filtered_corpus, filtered_stars = [], []
    
    for i in range(len(corpus)):
        if i in mids:
            next
        else:
            filtered_corpus.append(corpus[i]), filtered_stars.append(stars[i])
    
    return filtered_corpus, filtered_stars