"""
Pirated from everywhere
    Ignacio Arroyo
    Copyright
    RUN IN PYTHON-3 TO AVOID UNICODE WARNINGS
"""

import _pickle as pickle
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import RadiusNeighborsClassifier, KNeighborsClassifier
from sklearn.linear_model import SGDClassifier,LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
import logging, os
from six import iteritems
import argparse
import codecs

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

from collections import Counter
from sklearn.cross_validation import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.metrics import classification_report_imbalanced
from sklearn.datasets import load_files

from gensim import matutils
from gensim.models.lsimodel import LsiModel
from gensim.sklearn_api.lsimodel import LsiTransformer
from gensim.sklearn_api.ldamodel import LdaTransformer

from pdb import set_trace as st # Debug the program step by step calling st()

print(__doc__)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
mark = "%%%_"


class windowStreamer(object):
    def __init__(self, input_file, vectorizer, dictionary, wsize=10, word_context="c"):
        self.file_name = input_file
        #self.analyzer = vectorizer.build_analyzer()
        self.tokenizer = vectorizer.build_tokenizer()
        self.wsize = wsize
        self.dictionary = dictionary
        self.word_context = word_context

    def __iter__(self):
        for line in codecs.open(self.file_name, mode="r",
                                    encoding='latin-1', errors='replace'):
            ln = self.tokenizer(line.lower())
            try:
                for i, _ in enumerate(ln):
                    if self.word_context == "w":
                        try:
                        #word = mark + ln[i]
                            yield ln[i]

                        except KeyError:
                            continue
                    else:
                        start = min(len(ln[0:i]), self.wsize)
                        w = ln[i - start:i] + ln[i + 1:i + (self.wsize + 1)]
                    #s = " ".join(w)
                    #wi = [word] + self.tokenizer(" ".join(self.analyzer(s)))
                    #window = [word] + self.tokenizer(" ".join(self.analyzer(s)))
                    #bow = dictionary.doc2bow(wi)
                    #if len(w) < 2:
                        #stderr.write("%s\n" % wi)
                    #    continue

                    #yield bow
                    #yield " ".join(w), word
                        yield self.dictionary.doc2bow(w)

            except IndexError:
                break

class corpus_streamer(object):
    """ This Object streams the input raw text file row by row.
    """
    def __init__(self, file_name, dictionary=None, strings=None, tokenizer=True):
        self.file_name=file_name
        self.dictionary=dictionary
        self.strings=strings
        self.tokenizer=tokenizer

    def __iter__(self):
        for line in codecs.open(self.file_name, 'r', 'utf-8'):#open(self.file_name):
        # assume there's one document per line, tokens separated by whitespace
            if self.dictionary and not self.strings:
                yield self.dictionary.doc2bow(line.lower().split())
            elif not self.dictionary and self.strings:
                if not self.tokenizer:
                    yield line.strip().lower()
                if self.tokenizer:
                    yield list(self.tokenizer(line))

def save_obj(obj, filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
              raise
    with open(filename + '.pkl', 'wb') as f:
        pickle.dump(obj, f)#, pickle.HIGHEST_PROTOCOL)

def load_obj(filename):
    with open(filename + '.pkl', 'rb') as f:
        return pickle.load(f)

def load_dataset(win_lsa_file, win_tags_file, text=True, ratio=0.7,
                  windows="windows", f_min=1, f_max=0):

    if not text:
        labels={}
        i=0
        for line in codecs.open(win_tags_file, 'r', 'latin-1'):
            labels[i]=line.strip().lower()
            i+=1
        data=np.loadtxt(win_lsa_file)
        if not data.shape[0] == len(labels):
            print ("number of samples and number of labels do not match... EXIT.")
            exit()
        return train_test_split(data, labels, train_size = ratio)
    else:
        if not os.path.exists(windows):
            populations={}
            populations["/*max_freq*/"]=0
            i=0
            for lab, win in zip(#open(win_tags_file, 'r'),
                                codecs.open(win_tags_file, 'r', 'latin-1'),
                                #open(win_lsa_file, 'r')
                                codecs.open(win_lsa_file, 'r', 'latin-1')
                            ):
                if not lab.strip().isalpha():
                    continue

                filename = "%s/%s/%d" % (windows, lab.strip().lower(), i)
                if not os.path.exists(os.path.dirname(filename)):
                    try:
                        os.makedirs(os.path.dirname(filename))
                        populations[lab.strip().lower()]='1'
                    except OSError as exc: # Guard against race condition
                        if exc.errno != errno.EEXIST:
                            raise
                else:
                    try:
                        freq=int(populations[lab.strip().lower()])+1
                    except KeyError:
                        st()

                    populations[lab.strip().lower()] = str(freq)
                    if populations["/*max_freq*/"] < freq:
                        populations["/*max_freq*/"] = freq

                with codecs.open(filename, 'w', 'latin-1') as f:
                #with open(filename, 'w') as f:
                    f.write("%s" % win)

                i+=1

            save_obj(populations, windows + "_populations/sample_count")

        populations=load_obj(windows + "_populations/sample_count")
        taken_words=[k for k in populations
                        if (k != "/*max_freq*/" and k != "<number>" and
                            k != "<NUMBER>" and int(populations[k]) > f_min and
                            (int(populations[k]) < f_max if f_max > 0 else True)
                           )
                    ]

        all=load_files(windows, categories=taken_words)
        #(Pdb) all.keys() ['target_names', 'data', 'target', 'DESCR', 'filenames']
        #st()
        #print {n:(populations[n], l) for n, l in zip(all.target_names, all.target)}

        return train_test_split(all['data'], all['target'], train_size = ratio)

win_lsa_file="/almac/ignacio/data/windows/sample.doc"
win_tags_file="/almac/ignacio/data/windows/sample.i"

#win_tags_file="/almac/ignacio/swem/ngrams/outngrams/centers_AA.txt"
#win_lsa_file="/almac/ignacio/swem/ngrams/outngrams/all_AA.txt"
#win_lsa_file="/almac/ignacio/swem/ngrams/outngrams/500e3_AA.txt"
#win_tags_file="/almac/ignacio/swem/ngrams/outngrams/500e3c_AA.txt"
dataset="/almac/ignacio/data/windows/windows_"
f_min=10
f_max=1000

from gensim.corpora import Dictionary
vectorizer = TfidfVectorizer(
                          min_df=f_min,
                          encoding="latin-1",
                          decode_error="replace",
                          lowercase=True,
                          binary= True,# if args.tf.startswith("bin") else False,
                          sublinear_tf= False,# if args.tf.startswith("subl") else False,
                          stop_words= "english",# if args.stop else None
                        )
tokenizer = vectorizer.build_tokenizer()
#file_name, dictionary=None, strings=None, tokenizer=True
corpus = corpus_streamer("/home/iarroyof/Documentos/DERRL_es.txt", strings=True, tokenizer=tokenizer)
texts = [
    ['complier', 'system', 'computer'],
    ['eulerian', 'node', 'cycle', 'graph', 'tree', 'path'],
    ['graph', 'flow', 'network', 'graph'],
    ['loading', 'computer', 'system'],
    ['user', 'server', 'system'],
    ['tree', 'hamiltonian'],
    ['graph', 'trees'],
    ['computer', 'kernel', 'malfunction', 'computer'],
    ['server', 'system', 'computer']
]
dictionary = Dictionary(corpus)
#X_train = [dictionary.doc2bow(text) for text in texts]

#d_train = windowStreamer("/home/iarroyof/Documentos/DERRL_es.txt", vectorizer, dictionary, 5)
#X_train = [x for x, _ in d_train]
X_train = windowStreamer("/home/iarroyof/Documentos/DERRL_es.txt", vectorizer, dictionary, 5, "c")
#y_train = [y for _, y in d_train]#[:len(texts)]
y_train = [y for y in windowStreamer("/home/iarroyof/Documentos/DERRL_es.txt", vectorizer, dictionary, 5, "w")]

X_test = windowStreamer("/home/iarroyof/Documentos/DERRL_es.txt", vectorizer, dictionary, 5, "c")
y_test = [y for y in windowStreamer("/home/iarroyof/Documentos/DERRL_es.txt", vectorizer, dictionary, 5, "w")]
#d_test = windowStreamer("/home/iarroyof/Documentos/test.txt", vectorizer, dictionary, 5)
#X_test = (x for x, _ in d_test)
#X_test = [dictionary.doc2bow(text) for text in texts]
#y_test = [y for _, y in d_test]#[:len(texts)]

model = LsiTransformer(num_topics=2, id2word=dictionary)
#model = LdaTransformer(num_topics=15, id2word=dictionary, iterations=10, random_state=37)
#load_dataset(win_lsa_file, win_tags_file, text=True, ratio=0.7,
#                  windows="windows", f_min=1, f_max=0)
#X_train, X_test, y_train, y_test = load_dataset(win_lsa_file=win_lsa_file, win_tags_file=win_tags_file,
#                                                windows=dataset, f_min=f_min, f_max=f_max)

print('Training class distributions summary: {}'.format(Counter(y_train)))
print('Test class distributions summary: {}'.format(Counter(y_test)))

from sklearn.decomposition import KernelPCA
from sklearn.kernel_approximation import RBFSampler
from sklearn import linear_model

#pipe = make_pipeline_imb(
pipe = make_pipeline(model,
#                        TfidfVectorizer(
#                          min_df=f_min,
#                          encoding="latin-1",
#                          decode_error="replace",
#                          lowercase=True,
#                          binary= True,# if args.tf.startswith("bin") else False,
#                          sublinear_tf= False,# if args.tf.startswith("subl") else False,
#                          stop_words= "english",# if args.stop else None
#                        ),
                        #RandomOverSampler(),
#                        RandomUnderSampler(),
                        #SMOTEENN(random_state=0),
                        #SMOTETomek(random_state=42),
#                        Normalizer(),
#                        TruncatedSVD(200),
                        #KernelPCA(n_components=75, kernel="poly", gamma=10, degree=3, n_jobs=-1),
                        #KernelPCA( kernel="rbf", gamma=10, degree=3, n_jobs=-1),
                        #RBFSampler(gamma=0.1, random_state=1),
                        #SGDClassifier(alpha=.0001,
                        #              n_iter=100,
                        #              n_jobs=-1,
                        #              verbose=100,
                        #              epsilon=1,
                        #              class_weight='balanced',
                        #              #warm_start=True,
                        #              penalty='l1')
                        #GaussianProcessClassifier(n_jobs=-1)
                        linear_model.LogisticRegression(penalty='l2', C=0.1)
                        #MLPClassifier(hidden_layer_sizes=(75, 56), verbose=True, max_iter=300)
                        #LogisticRegression(C=5.0, verbose=100,
                        # multi_class='multinomial', n_jobs=-1, max_iter=100,
                        # penalty='l2', solver='saga', tol=0.1,class_weight='balanced')
                        #RadiusNeighborsClassifier(radius=2.0, weights='distance', algorithm='auto', leaf_size=10, p=2)
                        #KNeighborsClassifier(n_neighbors=1000, weights='distance', algorithm='brute', leaf_size=10, p=2, n_jobs=-1)
                        #LinearSVC(class_weight='balanced', verbose=100, max_iter=100, penalty='l1', dual=False)
                        #MultinomialNB()
                        )

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

print(classification_report_imbalanced(y_test, y_pred))
