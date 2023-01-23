#!/usr/bin/python
# -*- coding: latin-1 -*-
from pdb import set_trace as st

from gensim.models.keyedvectors import KeyedVectors as vDB
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import numbers
import argparse
import sys
import time

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
from pdb import set_trace as st
when_error = 0.5

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

def similarity(va, vb, d="cos"):
    if d.startswith("euc"):
        dp = np.linalg.norm(va - vb)
    elif d.startswith("man"):
        dp = np.absolute(va - vb).sum()
    else:# d.startswith("cos"):
        dp = np.dot(va, vb.T) / (np.linalg.norm(va) * np.linalg.norm(vb))

    return dp


def sts(i, pair, fo=None, dist='cos'):
    try:
        a, b = pair.split('\t')[:2]
    except IndexError:
        return i, "Entrada incompleta."

    try:
        va = series.transform(a)
        vb = series.transform(b)
    except TypeError:
        if not fo:
            pass
        else:
            fo.write("{:.4}\n".format(when_error))
        return i, "Error al inferir vector." # None
    
    try:
        sim = similarity(va, vb, dist)
        if fo:
            fo.write("{:.4}\n".format(sim))
        return i, sim
    except TypeError:
        if not fo:
            pass
        else:
            fo.write("{:.4}\n".format(when_error))
        return i, "Error al calcular similitud [TypeError]" # None

    except AttributeError:
        if not fo:
            pass
        else:
            fo.write("{:.4}\n".format(when_error))
        return i, "Error al calcular similitud [AttributeError]"
    else:
        return i, "Error desconocido al calcular similitud."


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
    parser.add_argument("--verbose", help = "Toggle verbose.",
                                                        action="store_true")
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
# ---------

    vectorizer = TfidfVectorizer(min_df = 1,
                ngram_range=(1, args.ngrams),
                #encoding = "latin-1",
                decode_error = "replace",
                lowercase = True,
                binary = True if args.localw.startswith("bin") else False,
                sublinear_tf = True if args.localw.startswith("subl") else False,
                stop_words = "english" if args.stop == 'ost' else None)

    start = time.time()
    if args.idfmodel.startswith("none") or args.idfmodel is None:
        if args.verbose:
            logging.info("The word embeddings will be combined unweighted.")
        tfidf = None
        seg = 1.0
    elif not os.path.isfile(args.idfmodel) and not args.idfmodel.startswith("local"):
        logging.info("IDF model file does not exist (EXIT):"
                "\n%s\n ..." % args.idfmodel)
        exit()

    elif os.path.isfile(args.idfmodel):
        if args.verbose:
            logging.info("Loading global TFIDF weights from: %s ..." % args.idfmodel)
        with open(args.idfmodel, 'rb') as f:
            if pyVersion == '2':
                vectorizer = pickle.load(f)
            else:
                vectorizer = pickle.load(f, encoding = 'latin-1')

        if args.tfidf.startswith("tfidf"):
            tfidf = True
        elif args.tfidf.startswith("idf"):
            tfidf = False
        seg = 60.0

    elif args.idfmodel.startswith("local"):
        if args.verbose:
            logging.info("The word embeddings will be combined and weighted.")
        if args.tfidf.startswith("tfidf"):
            tfidf = True
        elif args.tfidf.startswith("idf"):
            tfidf = False
        if args.verbose:
            logging.info("Fitting local TFIDF weights from: %s ..." % args.input)
        vectorizer.fit(pairs)
        seg = 1.0
# --------
    tfidf_name = "none_idf" if args.idfmodel is None or "none" in args.idfmodel.lower() else os.path.basename(args.idfmodel).split(".")[0]

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
            if args.verbose:
                logging.info("Loading word embeddings from: %s ..." % args.embedmodel)
            embedding = load_vectors(args.embedmodel, binary = True,
                                                        encoding = "latin-1")
        elif args.format.startswith("tex"):
            if args.verbose:
                logging.info("Loading word embeddings from: %s ..." % args.embedmodel)
            embedding = load_vectors(args.embedmodel, binary = False,
                                                        encoding = "latin-1")
        else:
            if args.verbose:
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
    if args.verbose:
        logging.info("Embedding sentences ...")
    global series
    series = wisse.wisse(embeddings=embedding, vectorizer=vectorizer, tf_tfidf=tfidf,
                         combiner=args.comb, return_missing=False, generate=True,
                         verbose=args.verbose)
    if output_name != '':
        fo = open(output_name, "w")
    else:
        fo = None
    if args.verbose:
        logging.info("Computing similarities...")

    if args.njobs > 1:
        similarities = Parallel(n_jobs=args.njobs)(delayed(sts)(i, pair, fo, metric)
                                            for i, pair in enumerate(pairs))
    else:
        similarities = []
        for i, pair in enumerate(pairs):
            similarities.append(sts(i, pair, fo, metric))

    if args.verbose:
        for i, s in similarities:
            if isinstance(s, numbers.Number):
                print("{:.4}".format(s))
            else:
                print("NONE")

logging.info("FINISHED! in %f %s. See output: %s \n" % ((time.time() - start)/seg,
                                                        'm' if seg != 1.0 else 's' ,
                                                        output_name))

