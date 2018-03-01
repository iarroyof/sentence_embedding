#!/usr/bin/python
# -*- coding: latin-1 -*-
# Python2.7

import numpy as np
import logging
import os
from functools import partial
from pdb import set_trace as st
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


class wisse(object):
    """ Both the TFIDFVectorizer and the word embedding model must be pretrained, either from the local 
        sentence corpus or from model persintence.
    """
    def __init__(self, embeddings, vectorizer, tf_tfidf, combiner = "sum"):
        self.tokenize = vectorizer.build_tokenizer()
        self.tfidf = vectorizer
        self.embedding = embeddings
        self.pred_tfidf = tf_tfidf
        if combiner.startswith("avg"):
            self.comb = partial(np.mean, axis = 0)
        else:
            self.comb = partial(np.sum, axis = 0)


    def fit(self, X, y = None): # Scikit-learn template
        if isinstance(X, list):
            self.sentences = X

        return self


    def transform(self, X):
        if isinstance(X, list):
            return self.fit(X)

        elif isinstance(X, str):
            return self.infer_sentence(X)

    
    def fit_transform(self, X, y=None):
        return self.transform(X)


    def infer_sentence(self, sent):
        ss = self.tokenize(sent)
        missing_bow = []
        missing_cbow = []
        series = {}

        if not ss == []:
            self.weights, m = self.infer_tfidf_weights(ss)
        else:
            return None

        missing_bow += m

        for w in self.weights:
            try:
                series[w] = (self.weights[w], self.embedding[w])
            except KeyError:
                series[w] = None
                missing_cbow.append(w)
                continue
            except IndexError:
                continue

        if self.weights == {}: return None
        # Embedding the sentence... :
        sentence = np.array([series[w][1] for w in series if not series[w] is None])
        series = {}

        return missing_cbow, missing_bow, self.comb(sentence)


    def infer_tfidf_weights(self, sentence):
        existent = {}
        missing = []

        if not self.tfidf:
            for word in sentence:
                existent[word] = 1.0

            return existent, missing

        if self.pred_tfidf:
            unseen = self.tfidf.transform([" ".join(sentence)]).toarray()
            for word in sentence:
                try:
                    existent[word] = unseen[0][self.tfidf.vocabulary_[word]]
                except KeyError:
                    missing.append(word)
                    continue
        else:
            for word in sentence:
                try:
                    weight = vectorizer.idf_[vectorizer.vocabulary_[word]]
                    existent[word] = weight if weight > 2 else 0.01
                except KeyError:
                    missing.append(word)
                    continue

        return existent, missing


    def __iter__(self):
        for s in self.sentences:
            yield self.transform(s)


def save_dense(directory, filename, array):
    directory=os.path.normpath(directory) + '/'
#    try:
    if filename.isalpha():
            np.save(directory + filename, array)
    else:
            return None
#    except UnicodeEncodeError:
#        return None    

def load_dense(filename):
    return np.load(filename)


def load_sparse_bsr(filename):
    loader = np.load(filename) 
    return bsr_matrix((loader['data'], loader['indices'], loader['indptr']),                       
        shape=loader['shape']) 


def save_sparse_bsr(directory, filename, array):     
# note that .npz extension is added automatically     
    directory=os.path.normpath(directory) + '/'
    if word.isalpha():
        array=array.tobsr()
        np.savez(directory + filename, data=array.data, indices=array.indices,              
            indptr=array.indptr, shape=array.shape) 
    else:
        return None


class vector_space(object):
    def __init__(self, directory, sparse = False):
        self.sparse = sparse 
        ext = ".npz" if sparse else ".npy"
        if directory.endswith(".tar.gz"):
            self._tar = True
            import tarfile
            self.tar = tarfile.open(directory)
            file_list = self.tar.getnames() #[os.path.basename(n) for n in self.tar.getnames()]
            self.words = {os.path.basename(word).replace(ext, ''): word 
                                                    for word in file_list}
        else:
            self._tar = False
            directory = os.path.normpath(directory) + '/' 
            file_list = os.listdir(directory)
            self.words = {word.replace(ext, ''): directory + word 
                                                for word in file_list}


    def __getitem__(self, item):
        if self.sparse:
            if self._tar:
                member = self.tar.getmember(self.words[item])
                word = self.tar.extractfile(member)
            else:
                word = self.words[item]
            #return load_sparse_bsr(self.words[item])
            return load_sparse_bsr(word) 

        else:
            if self._tar:
                member = self.tar.getmember(self.words[item])
                word = self.tar.extractfile(member)
            else:
                word = self.words[item]
            #return load_sparse_bsr(self.words[item])
            return load_dense(word)


def keyed2indexed(keyed_model, output_dir = "word_embeddings/", parallel = True, n_jobs = -1):
    output_dir = os.path.normpath(output_dir) + '/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if parallel:
        from joblib import Parallel, delayed

        Parallel(n_jobs = n_jobs, verbose = 10)(delayed(save_dense)(output_dir, word, keyed_model[word]) 
                                                        for word, _ in keyed_model.vocab.items())
    else:
        for word, _ in keyed_model.vocab.items():
            save_dense(output_dir, word, keyed_model[word])
    

class streamer(object):
    def __init__(self, file_name):
        self.file_name = file_name

    def __iter__(self):
        for s in open(self.file_name):
            yield s.strip()
