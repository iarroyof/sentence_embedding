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


def sts(i, pair):
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
        return i, similarity(va, vb, fo)
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
    parser.add_argument("--tfidf", help="To predict TFIDF complete weights "
                                        "('tfidf') or use only partial IDFs "
                                        "('idf'). (default = 'tfidf').",
                                        default = "tfidf")
    parser.add_argument("--localw", help = "TFIDF word vector weights "
                                    "computed locally from the input file of "
                                    "sentences {freq, binary, sublinear} "
                                    "(default='none').", default = "none")
    parser.add_argument("--stop", help = "Toggles stripping stop words in "
                                    "locally computed word vector weights. ",
                                                        action = "store_true")
    parser.add_argument("--format", help = "The format of the embedding model "
                                     "file: {binary, text, wisse}. "
                                    "default = 'binary'.", default = "binary")
    parser.add_argument("--ngrams", help = "The n-gram range specified as  "
                       "e.g. '(1-3)' for 1-grams, 2-grams and 3-grams, "
                       "considered to obtain TF-IDF weights. Default = '(1-1)'.", default = "(1-1)")
    args = parser.parse_args()


    if not args.format.startswith("wisse"):
        if not os.path.isfile(args.embedmodel):
            logging.info("Embedding model file does not exist (EXIT):"
                "\n%s\n ..." % args.embedmodel)
            exit()
        load_vectors = vDB.load_word2vec_format

    elif not os.path.exists(args.embedmodel):
        logging.info("Embedding model directory does not exist (EXIT):"
                "\n%s\n ..." % args.embedmodel)
        exit()


    if not os.path.isfile(args.idfmodel) and not args.idfmodel.startswith("local"):
        logging.info("IDF model file does not exist (EXIT):"
                "\n%s\n ..." % args.idfmodel)
        exit()
    if not os.path.isfile(args.input):
        logging.info("Input file does not exist (EXIT):"
                "\n%s\n ..." % args.input)
        exit()
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

    if args.tfidf.startswith("tfidf"):
        pred_tfidf = True
    elif args.tfidf.startswith("idf"):
        pred_tfidf = False
    else:
        pred_tfidf = False
        tfidf = False

    try:
        r_a, r_b = args.ngrams.replace('(', '').replace(')', '').split('-')
    except:
        print("ERROR: n-gram range must be specified by two integers either as "
        "'int_1-int_2' or as '(int_1-int_2)' or as int_1-int_2; "
        "without spaces. ")
        exit()
    vectorizer = TfidfVectorizer(min_df = 1,
     ngram_range=(int(r_a), int(r_b)),
                encoding = "latin-1",
                decode_error = "replace",
                lowercase = True,
                binary = True if args.localw.startswith("bin") else False,
                sublinear_tf = True if args.localw.startswith("subl") else False,
                stop_words = "english" if args.stop else None)

    pairs = wisse.streamer(args.input)

    if args.idfmodel.startswith("local"):
        logging.info("Fitting local TFIDF weights from: %s ..." % args.input)
        tfidf = vectorizer.fit(pairs)

    elif os.path.isfile(args.idfmodel):
        logging.info("Loading global TFIDF weights from: %s ..." % args.idfmodel)
        with open(args.idfmodel, 'rb') as f:
            if pyVersion == '2':
                tfidf = pickle.load(f)
            else:
                tfidf = pickle.load(f, encoding = 'latin-1')

    else:
        tfidf = False

    try:
        if args.format.startswith("bin"):
            embedding = load_vectors(args.embedmodel, binary = True,
                                                        encoding = "latin-1")
        elif args.format.startswith("tex"):
            embedding = load_vectors(args.embedmodel, binary = False,
                                                        encoding = "latin-1")
        else:
            embedding = wisse.vector_space(args.embedmodel, sparse = False)

    except:
        logging.info(
            """Error while loading word embedding model. Verify if the file
            is broken (EXIT)...\n%s\n""" % args.embedmodel)
        exit()

    embedding_name = os.path.basename(args.embedmodel).split(".")[0]
    tfidf_name = os.path.basename(args.idfmodel).split(".")[0]

    logging.info("\n\nEmbedding sentences ..\n%s\n" % output_name)
    global series
    series = wisse.wisse(embeddings = embedding, vectorizer = tfidf, 
                             tf_tfidf = True, combiner='sum', return_missing=False, generate=True)
    if output_name != '':
        fo = open(output_name, "w") 
    else:
        fo = None
        
    #incomplete=[]
    similarities = Parallel(n_jobs=1)(delayed(sts)(i, pair) 
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

logging.info("FINISHED! \n")

