import json
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np

# ==========================================
# 1. Load DNA sequences
# ==========================================

def load_sequences(json_path="random_sequences.json"):
    with open(json_path, "r") as f:
        data = json.load(f)
    # Return only sequences in a list
    sequences = [data[key]["sequence"] for key in data]
    return sequences

# ==========================================
# 2. Convert each sequence into k-mers
# ==========================================

def kmers(seq, k=6):
    if len(seq) < k:
        return [seq]
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]

def build_documents(sequences, k=6):
    docs = []
    for idx, seq in enumerate(sequences):
        tokens = kmers(seq, k)
        docs.append(TaggedDocument(words=tokens, tags=[str(idx)]))
    return docs

# ==========================================
# 3. Train Seq2Vec (Doc2Vec)
# ==========================================

def train_seq2vec(documents, vector_size=100):
    model = Doc2Vec(
        documents,
        vector_size=vector_size,
        window=5,
        min_count=1,
        workers=4,
        epochs=30,
        dm=0   # Kimothi used DBOW (Doc2Vec architecture)
    )
    return model

# ==========================================
# 4. Infer sequence embedding
# ==========================================

def sequence_embedding(model, seq, k=6):
    tokens = kmers(seq, k)
    return model.infer_vector(tokens)

# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    print("Loading DNA sequences...")
    sequences = load_sequences("random_sequences.json")

    print("Building documents for Doc2Vec...")
    documents = build_documents(sequences, k=6)

    print("Training Seq2Vec (Doc2Vec) model...")
    model = train_seq2vec(documents, vector_size=100)

    model.save("seq2vec_dna.model")
    print("Saved model as seq2vec_dna.model")

    print("\nSequence embeddings:")
    for i, seq in enumerate(sequences):
        emb = sequence_embedding(model, seq, k=6)
        print(f"seq{i+1} embedding shape: {emb.shape}")

    print("\nDone.")
