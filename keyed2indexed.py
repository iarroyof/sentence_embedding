import wisse
from gensim.models.keyedvectors import KeyedVectors as vDB
import sys
import logging

# sys.argv[1]: Input embeddings model (w2v format)
# sys.argv[2]: Output direcory for indexed format
# sys.argv[3]: Input format (default: binary)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.INFO)

load_vectors = vDB.load_word2vec_format

try:
    if sys.argv[3]:
        binary = False
except:
    binary = True

embedding = load_vectors(sys.argv[1], binary=binary, encoding = "latin-1")
logging.info("""Indexing embeddings, this will take a while...\n""")
wisse.keyed2indexed(embedding, sys.argv[2])
logging.info("""Embeddings indexed, please verify the contents of the output directory...\n""")
