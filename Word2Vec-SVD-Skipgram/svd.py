import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
import contractions
import re
from tqdm import tqdm
import numpy as np
import time
import torch
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import svds

def read_csv(file_path):
    df = pd.read_csv(file_path)
    text = df['Description'].tolist()
    labels = df['Class Index'].tolist()
    return text, labels

def tokenize(text):
    text = contractions.fix(text)
    sentences = sent_tokenize(text)
    new_sents = []
    for sentence in sentences:
        sentence = re.sub(r'[^\w\s]', '', sentence)
        sentence = re.sub(r'\b\d+\b', 'num', sentence)
        sentence = re.sub(r'\s+', ' ', sentence).strip()
        new_sents.append(sentence)
    tokenized_text = [['<start>']+word_tokenize(t)+['<end>'] for t in new_sents] 
    return tokenized_text
    
def word_frequency(text):
    word_dict={}
    for sample in text:
        for sentence in sample:
            for word in sentence:
                if word in word_dict:
                    word_dict[word]+=1
                else:
                    word_dict[word]=1
    return word_dict

def replace_low_frequency_words_with_unk(text,cut_off_frequency=3):
    word_dict = word_frequency(text)
    unk_words = set([word for word, frequency in word_dict.items() if frequency <= cut_off_frequency])
    out_text = []
    for sample in text:
        replaced_sentences = [[token if token not in unk_words else '<unk>' for token in sentence] for sentence in sample]
        out_text += replaced_sentences
    words = []
    for sentence in out_text:
        words += sentence
    return out_text, unk_words, set(words)

def get_word2idx(words):
    word2idx = {}
    for i,word in enumerate(words):
        word2idx[word] = i
    return word2idx

def create_co_occurance_matrix(text, word2idx, window_size=1):
    vocab_size = len(word2idx)
    co_occurrence_matrix = lil_matrix((vocab_size, vocab_size), dtype=np.float64)
    
    for sentence in text:
        for i, word in enumerate(sentence):
            center_word_idx = word2idx.get(word, None)
            start_index = max(0, i - window_size)
            end_index = min(len(sentence), i+window_size+1)
            context_words = sentence[start_index:i] + sentence[i+1:end_index]
            
            for context_word in context_words:
                context_word_idx = word2idx[context_word]
                if context_word_idx is not None:
                    co_occurrence_matrix[center_word_idx, context_word_idx] += 1
                    co_occurrence_matrix[context_word_idx, center_word_idx] += 1
    
    return co_occurrence_matrix

if __name__ == '__main__':
    print("Reading files")
    train_text, train_labels = read_csv('ANLP-2/train.csv')
    test_text, test_labels = read_csv('ANLP-2/test.csv')
    
    print("Pre-processing....")
    tokenised_text = [tokenize(sample) for sample in train_text]
    tokenized_text_unk, unk_words, uniq_words = replace_low_frequency_words_with_unk(tokenised_text)
    word2idx = get_word2idx(uniq_words)

    for window_size in [1,3,4,5]:
        
        print(f"window size {window_size}")
        print("Creating co-occurance matrix")
        co_occurance_matrix = create_co_occurance_matrix(tokenized_text_unk, word2idx, window_size=window_size)
        print("SVD computing")
        start_time = time.time()
        u, s, vh = svds(co_occurance_matrix, k=256)
        print(f"Time = {(time.time()-start_time)/60} minutes")
        u_copy = u.copy()
        word_vectors_tensor = torch.from_numpy(u_copy)
        file_path = "svd-word-vectors_"+str(window_size)+".pt"
        print("Saving as",file_path)
        torch.save((word2idx ,word_vectors_tensor), file_path)
