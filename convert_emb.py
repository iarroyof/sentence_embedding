from gensim.models import KeyedVectors

# Load the text embeddings
model = KeyedVectors.load_word2vec_format("dummy_embeddings.txt", binary=False)

# Save in binary format
model.save_word2vec_format("dummy_embeddings.bin", binary=True)

print("Converted dummy_embeddings.txt to dummy_embeddings.bin")
