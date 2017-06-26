from numpy import array, sum, dot, vstack, zeros
from numpy.linalg import norm
import argparse
import logging
import cPickle as pickle
from pdb import set_trace as st

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
level=logging.INFO)

class streamer(object):
    def __init__(self, file_name):
        self.file_name=file_name

    def __iter__(self):
        for s in open(self.file_name):
            yield s.strip().split()

def gm_0(input_file, output_file):
    vectors=streamer(input_file)
    basis = []
    types = []
    for v in vectors:
#        st()
        if len(v) == 2:
            with open(output_file, "wt") as f:
                f.write("%s\n" % " ".join(v))
            continue
        v_=array([float(x) for x in v[1:]])
        w = v_ - sum( dot(v_, b)*b  for b in basis )
        if (w > 1e-10).any():  
            basis.append(w/norm(w))
            types.append(v[0])

    with open(output_file, "at") as f:
        for w, v in zip(types, basis):
            f.write("%s %s\n" % (w, " ".join([str(r) for r in list(v)]) ) )

def gm_1(input_file, q_file, r_file):
    vectors=streamer(input_file)
    for v in vectors:
        if len(v) == 2:
            X=zeros((1, int(v[1]))).astype("float32")
            continue
        X=vstack([X, [float(x) for x in v[1:]]])
    X=delete(X, 0, 0)
    Q, R = np.linalg.qr(X)
    del X
    pickle.dump(Q, q_file)
    pickle.dump(R, r_file)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--embed", help="""Input file containing TFIDF pre-trained
                                            weights. If not provided, all weights
                                            will be 1.0 (pickled sklearn object).""",
                                            required=True)
    parser.add_argument("--ortho", help="""Output file containing word embeddings
                                            model (extension says me file type:
                                            binary or text).""", default="orthogonalized.vec")
    parser.add_argument("-A", help="""Output file containing word embeddings
                                            model (extension says me file type:
                                            binary or text).""", default="A.pk")
    
    args = parser.parse_args()
    logging.info("Fitting transformation weights from: %s ...\n" % args.embed)
    #gm_0(args.embed, args.ortho)
    gm_1(args.embed, args.ortho, args.A)
    logging.info("Saved: \n%s and:  %s\n ..." % (args.ortho,args.A) )
