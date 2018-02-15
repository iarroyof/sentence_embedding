from __future__ import print_function

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import cPickle as pickle
import argparse
import logging
from time import time
import numpy as np


class streamer(object):
    def __init__(self, file_name):
        self.file_name=file_name
    def __iter__(self):
        for s in open(self.file_name):
            yield s.strip()

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# Load some categories from the training set
categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]
# Uncomment the following to do the analysis on all the categories
# categories = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Computes Cross-Entropy (TFIDF) weights of a raw text dataset and stores the model.')
    parser.add_argument("--dataset", help="The path to the raw text dataset file",
                                                                required=True)
    parser.add_argument("--cout", help="The path to the cross-entropy output model file",
                                                                default="output_tfidf.pk")
    parser.add_argument("--minc", help="The minimum word frequency considered to compute CE weight.",
                                                                default=2, type=int)
    parser.add_argument("--tf", help="TF normalization: none, binary, sublinear (default=none).", default="none")
    parser.add_argument("--stop", help="Toggles stop words stripping.", action="store_true")
    parser.add_argument("--lsa", help="Toggles LSA computation.", default=0, type=int)
    parser.add_argument("--news", help="Toggles making analysis of predefined dataset.", action="store_true")
    args = parser.parse_args()
    t0 = time()
    if not args.news:
        corpus=streamer(args.dataset)
        vectorizer = TfidfVectorizer(min_df=1, 
                                 encoding="latin-1", 
                                 decode_error="replace", 
                                 lowercase=False, 
                                 binary= True if args.tf.startswith("bin") else False, 
                                 sublinear_tf= True if args.tf.startswith("subl") else False,
                                 stop_words= "english" if args.stop else None)

        X = vectorizer.fit(corpus) if args.lsa<0 else vectorizer.fit_transform(corpus)

    else:
        print("Loading 20 newsgroups dataset for categories:")
        print(categories)

        dataset = fetch_20newsgroups(subset='all', categories=categories,
                             shuffle=True, random_state=42)
        print("%d categories" % len(dataset.target_names))
        print()

        labels = dataset.target
        true_k = np.unique(labels).shape[0]

        print("%d documents" % len(dataset.data))
        vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,
                                 min_df=2, stop_words='english',
                                 use_idf=opts.use_idf)
        X = vectorizer.fit_transform(dataset.data)

    print("done in %fs" % (time() - t0))
    print("n_samples: %d, n_features: %d" % X.shape)
    print()

    if args.lsa==0:
        with open(args.cout, 'wb') as fin:
            pickle.dump(X, fin)
        print("TF-IDF weights saved...")
        exit()

    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import Normalizer
    from sklearn.pipeline import make_pipeline

    svd = TruncatedSVD(args.lsa)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    X = lsa.fit_transform(X)

    print("done in %fs" % (time() - t0))
    explained_variance = svd.explained_variance_ratio_.sum()
    print("Explained variance of the SVD step: {}%".format(
        int(explained_variance * 100)))

    print ("Saving vectors to: %s" % args.cout)
    np.savetxt(args.cout,X)


