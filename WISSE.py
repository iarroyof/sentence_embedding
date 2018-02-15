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

load_vectors = vDB.load_word2vec_format

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


class wisse(object):
    """ The TFIDFVectorizer must be pretrained, either from the local sentence corpus or
        from model persintence.
    """
    def __init__(self, embeddings, vectorizer, tf_tfidf):
        self.word2vec = word2vec
        self.tokenizer = vectorizer.build_tokenizer()
        self.tfidf = vectorizer
        self.embedding = embeddings
        self.pred_tfidf = tf_tfidf

    def fit(self, sent):
        return self.transform(sent)

    def transform(self, X):
        if isinstance(X, list):
            return wisse_streamer(X)
        
        return infer_sentence(X, self.tfidf, self.embedding, self.pred_tfidf)
    
    def fit_transform(self, X, y=None):
        return self.transform(X)


    def infer_sentence(self, sent, tfidf, embedding, pred_tfidf)
        ss = tokenize(sent)
        if not ss == []:
            weights, m = infer_tfidf_weights(ss, self.tfidf, predict = self.pred_tfidf)
        else:
            return None

        missing_bow += m

        for w in weights:
            try:
                series[w] = (weights[w], embedding[w])
            except KeyError:
                series[w] = None
                missing_cbow.append(w)
                continue
            except IndexError:
                continue

        if weights == {}: return None
        # Embedding the sentence... :
        sentence = np.array([series[w][1] for w in series if not series[w] is None])
        series = {}
        if args.comb.startswith("avg"):
            return missing_cbow, missing_bow, sentence.mean(axis = 0)
        else:
            return missing_cbow, missing_bow, sentence.sum(axis = 0)
                        

class transform_stream(object):
    def __init__(self, sent_list):
        self.sent_list = sent_list

    def __iter__(self):
        for s in self.sent_list:
            yield wisse.transform(s)


class streamer(object):
    def __init__(self, file_name):
        self.file_name = file_name

    def __iter__(self):
        for s in open(self.file_name):
            yield s.strip()


class iterable(object):
    def __init__(self, list_):
        self.list_ = list_
        self.__buffer__ = " ".join(self.list_)
        
    def __iter__(self):
        self.__buffer__ = " ".join(self.list_)
        yield " ".join(self.list_)


def infer_tfidf_weights(sentence, vectorizer, predict=False):
    existent = {}
    missing = []

    if not vectorizer:
        for word in sentence:
            existent[word] = 1.0

        return existent, missing

    if predict:
        #it = iterable(sentence)
        unseen = vectorizer.transform([" ".join(sentence)]).toarray()
        for word in sentence:
            try:
                existent[word] = unseen[0][vectorizer.vocabulary_[word]]
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
                                     file: {binary, text}. default = 'binary'""",
                                                            default = "binary")
    args = parser.parse_args()

    if not os.path.isfile(args.embedmodel):
        logging.info("""Embedding model file does not exist (EXIT):
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

    sentences = streamer(args.input)

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

    tokenize = vectorizer.build_tokenizer()

    if args.idfmodel.startswith("local"):
        logging.info("Fitting local TFIDF weights from: %s ..." % args.input)
        tfidf = vectorizer.fit(sentences)

    elif args.idfmodel is not None:
        logging.info("Loading global TFIDF weights from: %s ..." % args.idfmodel)
        with open(args.idfmodel, 'rb') as f:
            tfidf = pickle.load(f)#, encoding = 'latin-1')

    else:
        tfidf = False

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

    embedding_name = os.path.basename(args.embedmodel).split(".")[0]
    tfidf_name = os.path.basename(args.idfmodel).split(".")[0]

    missing_bow = []
    missing_cbow = []
    series = {}
    sidx = 1

    #if args.comb.startswith("avg"):
    #    combiner = partial(np.mean, axis = 0)
    #else:
    #    combiner = partial(np.sum, axis = 0)

    with open(output_name, "w") as fo:#, encoding = 'latin-1',
                                                    #errors = 'replace') as fo:
        for sent in sentences:
            ss = tokenize(sent)
            if not ss == []:
                weights, m = infer_tfidf_weights(ss, tfidf, predict = pred_tfidf)
            else:
                continue

            missing_bow += m

            for w in weights:
                try:
                    series[w] = (weights[w], embedding[w])
                except KeyError:
                    
                    series[w] = None
                    missing_cbow.append(w)
                    continue
                except IndexError:
                    continue

            if weights == {}: continue
            logging.info("Sentence weights %s" % weights)
            # Embedding the sentence... :
            sentence = np.array([series[w][1] for w in series if not series[w] is None])
            series = {}
            if args.comb.startswith("avg"):
                sentence = sentence.mean(axis = 0)
            #sentence = combiner(sentence)
            else:
                #sentence = ne.evaluate('sum(sentence,0)')
                sentence = sentence.sum(axis = 0)
                        
            fo.write("%d\t%s\n" % (sidx, np.array2string(sentence,
                                formatter = {'float_kind':lambda x: "%.6f" % x},
                                max_line_width = 20000).strip(']').strip('[') ))
            sidx += 1

    missing_name = (os.path.basename(args.input).split(".")[0] + "_" +
                                                        embedding_name + "_" +
                                                        tfidf_name + ".missing")
    with open(missing_name, "w") as f:
        f.write("# missing word embeddings:\n")
        for w in set(missing_cbow):
            f.write("%s\n" % w)

        f.write("# missing MI weights:\n")
        for w in set(missing_bow):
            f.write("%s\n" % w)

    logging.info("FINISHED! \n")
