
from sklearn.feature_extraction.text import TfidfVectorizer
import cPickle as pickle
import argparse
import logging

class streamer(object):
    def __init__(self, file_name):
        self.file_name=file_name
    def __iter__(self):
        for s in open(self.file_name):
            yield s.strip()

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Computes Cross-Entropy (TFIDF) weights of a raw text dataset and stores the model.')
    parser.add_argument("--dataset", help="The path to the raw text dataset file",
                                                                required=True)
    parser.add_argument("--cout", help="The path to the cross-entropy output model file",
                                                                default="output_tfidf.pk")
    parser.add_argument("--minc", help="The minimum word frequency considered to compute CE weight.",
                                                                default=2, type=int)
    parser.add_argument("--binary", help="Toggles binarize TF.", action="store_true")
    parser.add_argument("--sublinear", help="Toggles sublinear TF.", action="store_true")
    parser.add_argument("--stop", help="Toggles stop words stripping.", action="store_true")
    
    args = parser.parse_args()

    corpus=streamer(args.dataset)
    vectorizer = TfidfVectorizer(min_df=1, 
                                 encoding="latin-1", 
                                 decode_error="replace", 
                                 lowercase=False, 
                                 binary=args.binary, 
                                 sublinear_tf=args.sublinear,
                                 stop_words= "english" if args.stop else None)
    vectorizer
    tfidf = vectorizer.fit(corpus)
    with open(args.cout, 'wb') as fin:
        pickle.dump(tfidf, fin)
