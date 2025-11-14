import json
import gensim
from gensim.models import Word2Vec

# ==========================
# 1. Load sequences from JSON (generated earlier)
# ==========================

def load_sequences(json_path="random_sequences.json"):
    with open(json_path, "r") as f:
        data = json.load(f)
    sequences = [data[key]["sequence"] for key in data]
    return sequences

# ==========================
# 2. Build k-mers (Asgari method)
# ==========================

def kmers(seq, k=3):
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]

def build_corpus(sequences, k=3):
    corpus = []
    for seq in sequences:
        tokens = kmers(seq, k)
        corpus.append(tokens)
    return corpus

# ==========================
# 3. Train ProtVec model
# ==========================

def train_protvec(corpus, vector_size=100):
    model = Word2Vec(
        sentences=corpus,
        vector_size=vector_size,
        window=5,
        min_count=1,
        sg=1,                 # skip-gram
        workers=4,
        epochs=20
    )
    return model

# ==========================
# 4. Get sequence embeddings (Asgari averaged approach)
# ==========================

import numpy as np

def sequence_embedding(model, seq, k=3):
    tokens = kmers(seq, k)
    vectors = [model.wv[t] for t in tokens if t in model.wv]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

# ==========================
# MAIN — run Asgari ProtVec on generated DNA
# ==========================

if __name__ == "__main__":
    print("Loading generated DNA sequences...")
    sequences = load_sequences("random_sequences.json")

    print("Building corpus...")
    corpus = build_corpus(sequences, k=3)

    print("Training ProtVec (Asgari) model...")
    model = train_protvec(corpus, vector_size=100)

    model.save("protvec_dna.model")
    print("Saved model as protvec_dna.model")

    # compute embeddings for each sequence
    print("\nSequence embeddings:")
    for i, seq in enumerate(sequences, start=1):
        emb = sequence_embedding(model, seq, k=3)
        print(f"seq{i} embedding shape: {emb.shape}")

    print("\nDone.")
