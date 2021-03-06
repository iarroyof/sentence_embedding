#!/usr/bin/python
# -*- coding: latin-1 -*-
from pdb import set_trace as st

from gensim.models.keyedvectors import KeyedVectors as vDB
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import numbers
import argparse
import sys

pyVersion = sys.version.split()[0].split(".")[0]
if pyVersion == '2':
    import cPickle as pickle
else:
    import _pickle as pickle

import logging
import os
from functools import partial
import numpy as np
from joblib import Parallel, delayed
import wisse


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

def similarity(va, vb, file_pointer=None, d="cos"):
    if d.startswith("cos"):
        dp = np.dot(va, vb.T) / (np.linalg.norm(va) * np.linalg.norm(vb))
    elif d.startswith("euc"):
        dp = np.linalg.norm(va - vb)
    elif d.startswith("man"):
        dp = np.absolute(va - vb).sum()
        
    if file_pointer:
        file_pointer.write("{:.4}\n".format(dp))
        
    return dp


def sts(i, pair, fo=None, dist='cos'):
    try:
        a, b = pair.split('\t')[:2]
    except IndexError:
        return i, None
            
    try:
        va = series.transform(a)
        vb = series.transform(b)
    except TypeError:
        return i, None

    try:
        return i, similarity(va, vb, fo, dist)
    except TypeError:
        return i, None
            
    except AttributeError:
        return i, None


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This use example shows sentence "
        "embedding by using WISSE. The input is a text file which has a sentece in "
        "each of its rows. The output file has two tab-separated columns: the index "
        "line of the sentece in the input file and the sentence vector representation.")
    parser.add_argument("--idfmodel", help = "Input file containing IDF "
                                        "pre-trained weights. If not provided, "
                                        "all word vector weights will be set to "
                                        "1.0. If 'local' tf-idf weights will be "
                                        "computed locally from the input file "
                                        "(pickled sklearn object).",
                                        default = None)
    parser.add_argument("--embedmodel", help = "Input file containing word "
                                            "embeddings model (binary and text "
                                            "are allowed).", required = True)
    parser.add_argument("--output", help = "Output file containing the sentence "
                                            "embeddings.", default = "")
    parser.add_argument("--input", help = "Input file containing a sentence "
                                           "by row.", required = True)
    parser.add_argument("--comb", help = "Desired word vector combination for "
                                        "sentence representation {sum, avg}. "
                                        "(default = 'sum').", default = "sum")
    parser.add_argument("--suffix", nargs = '?', help = "A suffix to be added "
                                        "to the output file (default = '').",
                                            default = "", required = False)
    parser.add_argument("--tfidf", help="In local mode, to predict TFIDF complete weights "
                                        "('tfidf') or to use only partial IDFs "
                                        "('idf'). (default = 'tfidf').",
                                        default = "tfidf")
    parser.add_argument("--localw", help = "TFIDF word vector weights "
                                    "computed locally from the input file of "
                                    "sentences {freq, binary, sublinear} "
                                    "(default='none').", default = "none")
    parser.add_argument("--stop", help = "Stripping stop words ('ost') in "
                                    "locally computed word vector weights. "
                                    "Default='wst' (with, inlcuding, stop words)",    
                                                        default = "wst")
    parser.add_argument("--format", help = "The format of the embedding model "
                                     "file: {bin, txt, wisse}. "
                                    "default = 'bin'.", default = "bin")
    parser.add_argument("--dist", help = "The similarity metric. Available options: "
                                         " {cosine, euclidean, manhattan}. "
                                         "default = 'cosine'.", default = "cos")                                    
    parser.add_argument("--ngrams", help = "The n-gram limit specified as, "
                       "e.g., 3 for 1-grams, 2-grams and 3-grams, "
                       "considered to obtain TF-IDF weights. Default = 1.", 
                       default = 1, type=int)
    parser.add_argument("--njobs", help = "The number of jobs to compute "
                           "similarities of the input sentences, Default = 1.",
                           default = 1, type=int)
    args = parser.parse_args()


    if not os.path.isfile(args.input):
        logging.info("Input file can't be found. Impossible to continue (EXIT): "
                        "%s\n" % args.input)
        exit()
    else:
        pairs = wisse.streamer(args.input)

    if not args.format.startswith("wisse") and (args.format.startswith("bin") or args.format.startswith("txt") ):
        if not os.path.isfile(args.embedmodel):
            logging.info("Embedding model file does not exist (EXIT):"
                "\n%s\n ..." % args.embedmodel)
            exit()
        load_vectors = vDB.load_word2vec_format

    elif not os.path.exists(args.embedmodel) and args.format.startswith("wisse"):
        logging.info("Embedding model directory does not exist (EXIT):"
                "\n%s\n ..." % args.embedmodel)
        exit()
    elif not os.path.exists(args.embedmodel) and not args.format.startswith("wisse"):
        logging.info("Bad input format specification (EXIT): {bin, txt, wisse} "
                        "%s\n ..." % args.format)
        exit()

    vectorizer = TfidfVectorizer(min_df = 1,
                ngram_range=(1, args.ngrams),
                encoding = "latin-1",
                decode_error = "replace",
                lowercase = True,
                binary = True if args.localw.startswith("bin") else False,
                sublinear_tf = True if args.localw.startswith("subl") else False,
                stop_words = "english" if args.stop == 'ost' else None)
                
    if args.idfmodel.startswith("none"):
        logging.info("The word embeddings will be combined unweighted.")
        tfidf = False
    elif not os.path.isfile(args.idfmodel) and not args.idfmodel.startswith("local") and not args.idfmodel.startswith("none"):
        logging.info("IDF model file does not exist (EXIT):"
                "\n%s\n ..." % args.idfmodel)
        exit()
        
    elif os.path.isfile(args.idfmodel) and not args.idfmodel.startswith("local"):
        pred_tfidf = False
        logging.info("Loading global TFIDF weights from: %s ..." % args.idfmodel)
        with open(args.idfmodel, 'rb') as f:
            if pyVersion == '2':
                tfidf = pickle.load(f)
            else:
                tfidf = pickle.load(f, encoding = 'latin-1')
                
    elif args.idfmodel.startswith("local"):
        logging.info("The word embeddings will be combined and weighted.")
        tfidf = True
        if args.tfidf.startswith("tfidf") and tfidf:
            pred_tfidf = True
        elif args.tfidf.startswith("idf") and tfidf:
            pred_tfidf = False
            
        logging.info("Fitting local TFIDF weights from: %s ..." % args.input)
        tfidf = vectorizer.fit(pairs)

    if args.output != "" and args.output != "stdout":
        if os.path.dirname(args.output) != "":
            if not os.path.exists(os.path.dirname(args.output)):
                logging.info("Output directory does not exist (EXIT):"
                   "\n%s\n ..." % args.output)
                exit()
            else:
                output_name = args.output
        else:
            output_name = args.output
    elif args.output != "stdout":
        embed_name = os.path.abspath(args.embedmodel)
        suffix = "_".join([embed_name.split('/')[-1],
            args.comb,
            args.tfidf,
            "local" if args.idfmodel.startswith("local") else tfidf_name,
            args.suffix]).strip("_")
        output_name = args.input + ".output_" + suffix

    else:
        output_name = ''

    try:
        if args.format.startswith("bin"):
            logging.info("Loading word embeddings from: %s ..." % args.embedmodel)
            embedding = load_vectors(args.embedmodel, binary = True,
                                                        encoding = "latin-1")
        elif args.format.startswith("tex"):
            logging.info("Loading word embeddings from: %s ..." % args.embedmodel)
            embedding = load_vectors(args.embedmodel, binary = False,
                                                        encoding = "latin-1")
        else:
            logging.info("Loading word embeddings index from: %s ..." % args.embedmodel)
            embedding = wisse.vector_space(args.embedmodel, sparse = False)

    except:
        logging.info(
            "Error while loading word embedding model. Verify if the file "
            "is broken (EXIT)...\n%s" % args.embedmodel)
        exit()
        
    dss = ["cosine", "euclidean", "manhattan"]
    if not any([d.startswith(args.dist) for d in dss]):
        logging.info("Badly specified similarity metric: %s" 
                        "... setting the default (cosine)." % args.dist)
        metric = "cosine"
    else:
        metric = args.dist

    embedding_name = os.path.basename(args.embedmodel).split(".")[0]
    tfidf_name = os.path.basename(args.idfmodel).split(".")[0]

    logging.info("Embedding sentences ...")
    global series
    series = wisse.wisse(embeddings=embedding, vectorizer=tfidf, tf_tfidf=True, 
                            combiner=args.comb, return_missing=False, generate=True)
    if output_name != '':
        fo = open(output_name, "w") 
    else:
        fo = None
        
    logging.info("Computing similarities...")
    similarities = Parallel(n_jobs=args.njobs)(delayed(sts)(i, pair, fo, metric) 
                                            for i, pair in enumerate(pairs))
    #for i, pair in enumerate(pairs):
    for i, s in similarities:
        if isinstance(s, numbers.Number):
            print("{:.4}".format(s))
        else:
            print(" ")

            # At this point you can use the embeddings 'va' and 'vb' for any application 
            # as it is a numpy array. Also you can simply save the vectors in text format 
            # as follows:

logging.info("FINISHED! see output: %s \n" % output_name)
