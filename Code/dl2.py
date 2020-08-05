import numpy as np
import pandas as pd
import sys
import re
import pickle
from collections import defaultdict

#***********************************************#
#       word2vec dataset loader 2               #
#   Authors: Justin Weigle                      #
#   Edited: 25 Sep 2019                         #
#   Sources: github.com/yoonkim/CNN_sentence    #
#***********************************************#

#
# Load data and clean it
#
def build_data(dataloc, folds, cleanup):
    tweets = []
    hs_file = dataloc[0]
    ofn_file = dataloc[1]
    nei_file = dataloc[2]
    vocab = defaultdict(float)
    with open(hs_file, "rb") as f:
        for line in f:
            tweet = []
            tweet.append(line.strip())
            if cleanup:
                orig_tweet = cleanstr(b" ".join(tweet))
            else:
                orig_tweet = b" ".join(tweet).lower()
            words = set(orig_tweet.split())
            for word in words:
                vocab[word] += 1
            datum = {"y":0,
                     "text": orig_tweet,
                     "num_words": len(orig_tweet.split()),
                     "split": np.random.randint(0, folds)}
            tweets.append(datum)
    with open(ofn_file, "rb") as f:
        for line in f:
            tweet = []
            tweet.append(line.strip())
            if cleanup:
                orig_tweet = cleanstr(b" ".join(tweet))
            else:
                orig_tweet = b" ".join(tweet).lower()
            words = set(orig_tweet.split())
            for word in words:
                vocab[word] += 1
            datum = {"y":1,
                     "text": orig_tweet,
                     "num_words": len(orig_tweet.split()),
                     "split": np.random.randint(0, folds)}
            tweets.append(datum)
    with open(nei_file, "rb") as f:
        for line in f:
            tweet = []
            tweet.append(line.strip())
            if cleanup:
                orig_tweet = cleanstr(b" ".join(tweet))
            else:
                orig_tweet = b" ".join(tweet).lower()
            words = set(orig_tweet.split())
            for word in words:
                vocab[word] += 1
            datum = {"y":2,
                     "text": orig_tweet,
                     "num_words": len(orig_tweet.split()),
                     "split": np.random.randint(0, folds)}
            tweets.append(datum)
    #print(vocab)
    return tweets, vocab

#
# Loads 300x1 word vecs from word2vec
#
def load_bin_vec(fname, vocab):
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == b' ':
                    word = b''.join(word)
                    break
                if ch != b'\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.frombuffer(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs
#
# For words that occur in at least minimum documents, create a separate
# word vector.
# 0.25 used so unknown vectors have similar variance to pre-trained ones
#
def add_unknown_words(word_vecs, vocab, minimum = 1, k = 300):
    for word in vocab:
        if word not in word_vecs and vocab[word] >= minimum:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)

#
# Tokenization/string cleaning
#
def cleanstr(string):
    string = re.sub(br"[^A-Za-z0-9(),!?\'\`]", b" ", string)     
    string = re.sub(br"\'s", b" \'s", string) 
    string = re.sub(br"\'ve", b" \'ve", string) 
    string = re.sub(br"n\'t", b" n\'t", string) 
    string = re.sub(br"\'re", b" \'re", string) 
    string = re.sub(br"\'d", b" \'d", string) 
    string = re.sub(br"\'ll", b" \'ll", string) 
    string = re.sub(br",", b" , ", string) 
    string = re.sub(br"!", b" ! ", string) 
    string = re.sub(br"\(", b" \( ", string) 
    string = re.sub(br"\)", b" \) ", string) 
    string = re.sub(br"\?", b" \? ", string) 
    string = re.sub(br"\s{2,}", b" ", string)    
    return string.strip().lower()

#
# Get word matrix
#
def get_W(word_vecs, k = 300):
    vocab_size = len(word_vecs)
    word_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype = 'float32')
    W[0] = np.zeros(k, dtype = 'float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_map[word] = i
        i += 1
    return W, word_map


if __name__=="__main__":
    w2v_file = sys.argv[1]
    dataloc = ["../Datasets/HATE_SPEECH/hs.csv", "../Datasets/HATE_SPEECH/ofn.csv", "../Datasets/HATE_SPEECH/nei.csv"]
    print("Loading data...")
    tweets, vocab = build_data(dataloc, folds = 10, cleanup = True)
    max_l = np.max(pd.DataFrame(tweets)["num_words"])
    print("Data Loaded!")
    print("Number of sentences: " + str(len(tweets)))
    print("Vocab size: " + str(len(vocab)))
    print("Max sentence length: " + str(max_l))
    print("Loading word2vec vectors...")
    w2v = load_bin_vec(w2v_file, vocab)
    print("word2vec loaded!")
    print("Number of words already in word2vec: " + str(len(w2v)))
    add_unknown_words(w2v, vocab)
    W, word_map = get_W(w2v)
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab)
    W2, _ = get_W(rand_vecs)
    pickle.dump([tweets, W, W2, word_map, vocab], open("hst.p", "wb"))
    print("Dataset created successfully!!")
