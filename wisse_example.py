#!/usr/bin/python
# -*- coding: latin-1 -*-
# Python2.7
from gensim.models.keyedvectors import KeyedVectors as vDB
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import numexpr as ne
import argparse
#import _pickle as pickle
import cPickle as pickle
import logging
import os
from functools import partial
import wisse

from pdb import set_trace as st

load_vectors = vDB.load_word2vec_format

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="""This use example shows sentence 
        embedding by using WISSE. The input is a text file which has a sentece in 
        each of its rows. The output file has two tab-separated columns: the index
        line of the sentece in the input file and the sentence vector representation
        .""")
    parser.add_argument("--idfmodel", help = """Input file containing IDF
                                        pre-trained weights. If not provided,
                                        all word vector weights will be set to
                                        1.0. If 'local' tf-idf weights will be
                                        computed locally from the input file
                                        (pickled sklearn object).""",
                                        default = None)
    parser.add_argument("--embedmodel", help = """Input file containing word
                                            embeddings model (binary and text
                                            are allowed).""", required = True)
    parser.add_argument("--output", help = """Output file containing the sentence
                                            embeddings.""", default = "")
    parser.add_argument("--input", help = """Input file containing a sentence
                                            by row.""", required = True)
    parser.add_argument("--comb", help = """Desired word vector combination for
                                        sentence representation {sum, avg}.
                                        (default = 'sum')""", default = "sum")
    parser.add_argument("--suffix", nargs = '?', help = """A suffix to be added
                                        to the output file (default = '')""",
                                            default = "", required = False)
    parser.add_argument("--tfidf", help="""To predict TFIDF complete weights
                                        ('tfidf') or use only partial IDFs
                                        ('idf'). (default = 'tfidf')""",
                                        default = "tfidf")
    parser.add_argument("--localw", help = """TFIDF word vector weights
                                    computed locally from the input file of
                                    sentences {freq, binary, sublinear}
                                    (default='none').""", default = "none")
    parser.add_argument("--stop", help = """Toggles stripping stop words in
                                    locally computed word vector weights.""",
                                                        action = "store_true")
    parser.add_argument("--format", help = """The format of the embedding model
                                     file: {binary, text, wisse}. 
                                    default = 'binary'""", default = "binary")
    args = parser.parse_args()


    if not args.format.startswith("wisse"):
        if not os.path.isfile(args.embedmodel):
            logging.info("""Embedding model file does not exist (EXIT):
                \n%s\n ...""" % args.embedmodel)
            exit()
    elif not os.path.exists(args.embedmodel):
        logging.info("""Embedding model directory does not exist (EXIT):
                \n%s\n ...""" % args.embedmodel)
        exit()

    if not os.path.isfile(args.idfmodel) and not args.idfmodel.startswith("local"):
        logging.info("""IDF model file does not exist (EXIT):
                \n%s\n ...""" % args.idfmodel)
        exit()
    if not os.path.isfile(args.input):
        logging.info("""Input file does not exist (EXIT):
                \n%s\n ...""" % args.input)
        exit()
    if args.output != "":
        if os.path.dirname(args.output) != "":
            if not os.path.exists(os.path.dirname(args.output)):
                logging.info("""Output directory does not exist (EXIT):
                    \n%s\n ...""" % args.output)
                exit()
        else:
            output_name = args.output
    else:
        suffix = "_".join([embedding_name,
            args.comb,
            args.tfidf,
            "local" if args.idfmodel.startswith("local") else tfidf_name,
            args.suffix]).strip("_")
        output_name = args.input + ".output_" + suffix


    if args.tfidf.startswith("tfidf"):
        pred_tfidf = True
    elif args.tfidf.startswith("idf"):
        pred_tfidf = False
    else:
        pred_tfidf = False
        tfidf = False
    
    vectorizer = TfidfVectorizer(min_df = 1,
                encoding = "latin-1",
                decode_error = "replace",
                lowercase = True,
                binary = True if args.localw.startswith("bin") else False,
                sublinear_tf = True if args.localw.startswith("subl") else False,
                stop_words = "english" if args.stop else None)

    sentences = wisse.streamer(args.input)

    if args.idfmodel.startswith("local"):
        logging.info("Fitting local TFIDF weights from: %s ..." % args.input)
        tfidf = vectorizer.fit(sentences)

    elif os.path.isfile(args.idfmodel):
        logging.info("Loading global TFIDF weights from: %s ..." % args.idfmodel)
        with open(args.idfmodel, 'rb') as f:
            tfidf = pickle.load(f)#, encoding = 'latin-1')

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

    missing_bow = []    # Stores missing words in the TFIDF model
    missing_cbow = []   # Stores missing words in the W2V model
    sidx = 0 # The index of the sentence according to the input file
    logging.info("\n\nEmbedding sentences and saving then to a the output file..\n\n")

    with open(output_name, "w") as fo:
        for sent in sentences:
            sidx += 1
            series = wisse.wisse(embeddings = embedding, vectorizer = tfidf, 
                                                tf_tfidf = True, combiner='sum')
            try:
                mc, mb, vector = series.transform(sent)
            except TypeError:
                continue

            missing_cbow += mc
            missing_bow += mb
            fo.write("%d\t%s\n" % (sidx, np.array2string(vector,
                                formatter = {'float_kind':lambda x: "%.6f" % x},
                                max_line_width = 20000).strip(']').strip('[') ))

    missing_name = (os.path.basename(args.input).split(".")[0] + "_" +
                                                        embedding_name + "_" +
                                                        tfidf_name + ".missing")
    logging.info("\n\nSaving missing vocabulary to %s ..\n\n" % missing_name)

    with open(missing_name, "w") as f:
        f.write("# missing word embeddings:\n")
        for w in set(missing_cbow):
            f.write("%s\n" % w)

        f.write("# missing MI weights:\n")
        for w in set(missing_bow):
            f.write("%s\n" % w)

    logging.info("FINISHED! \n")
