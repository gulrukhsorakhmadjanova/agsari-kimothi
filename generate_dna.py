import random
import json

def generate_dna_sequence(min_len=100, max_len=1000):
    """Generate a random DNA sequence of random length."""
    length = random.randint(min_len, max_len)
    bases = ["A", "C", "G", "T"]
    seq = "".join(random.choice(bases) for _ in range(length))
    return seq, length

def save_fasta(sequences, filename="random_sequences.fasta"):
    """Save sequences to FASTA format."""
    with open(filename, "w") as f:
        for i, (seq, length) in enumerate(sequences, start=1):
            f.write(f">seq{i}_len_{length}\n{seq}\n")

def save_metadata(sequences, filename="random_sequences.json"):
    """Save sequence metadata to JSON for easy reuse."""
    data = {}
    for i, (seq, length) in enumerate(sequences, start=1):
        data[f"seq{i}"] = {
            "length": length,
            "sequence": seq
        }
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    # Generate two reusable DNA sequences
    seq1, len1 = generate_dna_sequence()
    seq2, len2 = generate_dna_sequence()

    sequences = [(seq1, len1), (seq2, len2)]

    # Save in FASTA format
    save_fasta(sequences)

    # Save detailed metadata (reusable format)
    save_metadata(sequences)

    print("Generated and saved:")
    print(f" - {len1}-bp sequence (seq1)")
    print(f" - {len2}-bp sequence (seq2)")
    print("\nFiles saved:")
    print("  random_sequences.fasta")
    print("  random_sequences.json")
