from numpy import array, einsum, arccos, newaxis, zeros, vstack
from statistics import  median, mean
from numpy.linalg import norm
from numpy import triu_indices
import argparse, logging
import cPickle as pickle
from gensim.models.keyedvectors import KeyedVectors as vDB
from pdb import set_trace as st

load_vectors=vDB.load_word2vec_format

class streamer(object):
    def __init__(self, file_name):
        self.file_name=file_name

    def __iter__(self):
        for s in open(self.file_name):
            yield s.strip().split("\t")[0].split()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--embed", help="""Input file containing pretrained word embeddings.""", required=True)
    parser.add_argument("--out", help="""Output file where to write angles and statistics.""", default=None)
    parser.add_argument("--sents", help="""Input file containing a document sentence by row.""", required=True)
    parser.add_argument("--amount", help="""Amount of inputs to process.""", default=10, type=int)
    parser.add_argument("--bin", help="""Binary (word2vec only) or text emebdding format.""", action="store_true")
    args = parser.parse_args()
    logging.info("Fitting distances from: %s ...\n" % args.embed)

    sents=streamer(args.sents)
    embedding=load_vectors(args.embed, binary=args.bin, encoding="latin-1")
    c=0
    if args.out:
        fo=open(args.out, "wb")
    
    for s in sents:
        if c >= args.amount:
            break
        sl=len(s) # sentece length
        X=[]
        for w in set(s):
            try:
                e = embedding[w]
                X.append(e)
            except KeyError:
                #print ("Word OOv %s\n" % w)
                continue
            except:
                print("No key error nut other stoped the program.")
                exit()
        X=array(X)
        
#X=array([[1.0,0.0], [1.0,1.0], [0.0,1.0], [-1.0, 2.0], [-1.0, -2.0], [-1.0, 0.0]])
        iu2 = triu_indices(X.shape[0], 1)

        dotprod_mat=einsum('ij,kj->ik', X, X)
        costheta = dotprod_mat / norm(X, axis=1)[:, newaxis]
        costheta /= norm(X, axis=1)
        #logging.info("Saved: \n%s and:  %s\n ..." % args.angles)
        angles=arccos(costheta)[iu2]
        
        logging.info("Computed angles sentence %d ..." % c)
        if not args.out:
            print ("%d %0.4f %0.4f %0.4f %0.4f %s" % (sl, angles.mean(), 
                                                   median(angles), 
                                                   angles.max(), 
                                                   angles.min(), 
                                                   angles.tolist()))
        else:
            fo.write("%d\t%0.4f\t%0.4f\t%0.4f\t%0.4f\t%s\n" % (sl, angles.mean(), 
                                                                median(angles), 
                                                                angles.max(), 
                                                                angles.min(), 
                                                                str(angles.tolist())[1:]\
                                                                    .strip("]")\
                                                                    .replace(", "," ") ) )
        c+=1
