#!/usr/bin/python
# -*- coding: latin-1 -*-
from gensim.models.keyedvectors import KeyedVectors as vDB
from sklearn.feature_extraction.text import TfidfVectorizer
from w2v import *
import numpy as np
import argparse
import _pickle as pickle
import logging


load_vectors = vDB.load_word2vec_format

from pdb import set_trace as st

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


class streamer(object):
    def __init__(self, file_name):
        self.file_name = file_name

    def __iter__(self):
        for s in open(self.file_name):
            yield s.strip()


def infer_tfidf_weights(sentence, vectorizer, predict=False):
    existent = {}
    missing = []

    if not vectorizer:
        for word in sentence.split():
            existent[word] = 1.0

        return existent, missing

    if predict:
        unseen = vectorizer.transform([sentence]).toarray()
        for word in sentence.strip().split():
            try:
                existent[word] = unseen[0][vectorizer.vocabulary_[word]]
            except KeyError:
                missing.append(word)
                continue
    else:
        for word in sentence.strip().split():
            try:
                weight = vectorizer.idf_[vectorizer.vocabulary_[word]]
                existent[word] = weight if weight > 2 else 0.01
            except KeyError:
                missing.append(word)
                continue

    return existent, missing


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
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
                                     file: {binary, text}. default = 'binary'""",
                                                            default = "binary")
    args = parser.parse_args()

    if not os.path.isfile(args.embedmodel):
        logging.info("""Embedding model file does not exist (EXIT):
                \n%s\n ...""" % args.embedmodel)
        exit()
    if not os.path.isfile(args.idfmodel):
        logging.info("""IDF model file does not exist (EXIT):
                \n%s\n ...""" % args.idfmodel)
        exit()
    if not os.path.isfile(args.input):
        logging.info("""Input file does not exist (EXIT):
                \n%s\n ...""" % args.input)
        exit()

    sentences = streamer(args.input)

    if args.tfidf.startswith("tfidf"):
        pred_tfidf = True
    elif args.tfidf.startswith("idf"):
        pred_tfidf = False
    else:
        pred_tfidf = False
        tfidf = False

    if args.idfmodel.startswith("local"):
        sentences = streamer(args.input)
        vectorizer = TfidfVectorizer(min_df = 1,
                encoding = "latin-1",
                decode_error = "replace",
                lowercase = False,
                binary = True if args.localw.startswith("bin") else False,
                sublinear_tf = True if args.localw.startswith("subl") else False,
                stop_words = "english" if args.stop else None)

        logging.info("Fitting local TFIDF weights from: %s ..." % args.input)
        tfidf = vectorizer.fit(sentences)
    elif args.idfmodel is not None:
        with open(args.idfmodel, 'rb') as f:
            tfidf = pickle.load(f, encoding = 'latin-1')

    else:
        tfidf = False

    logging.info("Loading word embedding model from:\n%s\n" % args.embedmodel)
    try:
        if args.format.startswith("bin"):
            embedding = load_vectors(args.embedmodel, binary = True,
                                                        encoding = "latin-1")
        else:
            embedding = load_vectors(args.embedmodel, binary = False,
                                                        encoding = "latin-1")
    except:
        logging.info(
            """Error while loading word embedding model. Verify if the file
            is broken (EXIT)...\n%s\n""" % args.embedmodel)
        exit()

    embedding_name = basename(args.embedmodel).split(".")[0]
    tfidf_name = basename(args.idfmodel).split(".")[0]
    suffix = "_".join([embedding_name,
            args.comb,
            args.tfidf,
            "local" if args.idfmodel.startswith("local") else tfidf_name,
            args.suffix]).strip("_")

    missing_bow = []
    missing_cbow = []
    series = {}
    sidx = 1

    if comb.startswith("avg"):
        combiner = np.mean
    else:
        combiner = np.sum
    combiner.axis = 0

    with open(args.input + ".output_" + suffix, "w", encoding = 'latin-1',
                                                    errors = 'replace') as fo:
        for sent in sentences:
            weights, m = infer_tfidf_weights(' '.join(clean_Ustring_fromU(sent)),
                                                    tfidf, predict=pred_tfidf)
            missing_bow += m
            for w in weights:
                try:
                    series[w] = (weights[w], embedding[w])
                except KeyError:
                    series[w] = None
                    missing_cbow.append(w)
                    continue

            logging.info("Sentence weights %s" % [(w, series[w][0]) for w in series
                                                    if not series[w] is None])

            series = array([series[w][1] for w in series if not series[w] is None])
            sentence = combiner(series).reshape(1, -1)

            fo.write("%d\t%s\n" % (sidx, np.array2string(sentence,
                                formatter = {'float_kind':lambda x: "%.6f" % x},
                                max_line_width = 20000).strip(']').strip('[') ))
            sidx += 1

    missing_name = (basename(args.input).split(".")[0] + "_" +
                                                        embedding_name + "_" +
                                                        tfidf_name + ".missing")

    with open(missing_name, "w") as f:
        f.write("# missing word embeddings:\n")
        for w in missing_cbow:
            f.write("%s\n" % w)

        f.write("# missing MI weights:\n")
        for w in missing_bow:
            f.write("%s\n" % w)

    logging.info("FINISHED! \n")
