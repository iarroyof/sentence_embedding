import wisse
from gensim.models.keyedvectors import KeyedVectors as vDB
import argparse
import logging

# sys.argv[1]: Input embeddings model (w2v format)
# sys.argv[2]: Output direcory for indexed format
# sys.argv[3]: Input format (default: binary)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.INFO)

load_vectors = vDB.load_word2vec_format

parser = argparse.ArgumentParser()
parser.add_argument("--input", help = "Input embeddings model (w2v format)", 
                                    required = True)
parser.add_argument("--output", help = "Output direcory for indexed format", 
                                    default = 'output_indexed')
parser.add_argument("--txt", help = "Toggles text word2vec format input format "
                                    "(default: binary)", 
                                    action='store_false')
args = parser.parse_args()                                 

binary = args.binary
embedding = load_vectors(args.input, binary=binary, encoding = "latin-1")
logging.info("Indexing embeddings, this will take a while...\n")

wisse.keyed2indexed(embedding, args.output)

logging.info("Embeddings indexed, please verify the contents of the output "
                "directory:\n %s\n" % args.output)
