import pickle
from collections import defaultdict, Counter
from Bio import SeqIO
import time


def defaultdict_int():
    """Returns a defaultdict with int as the default factory, replacing lambda."""
    return defaultdict(int)


def build_5mer_model(fasta_file):
    """
    Builds a 5-mer model from a FASTA file, including probabilities for starting 4-mers.

    Parameters:
    - fasta_file: Path to the FASTA file.

    Returns:
    - A dictionary representing the 5-mer model.
    - A Counter object for starting 4-mer counts.
    - A dictionary of starting 4-mer probabilities.
    """
    model = defaultdict(defaultdict_int)
    start_4mer_counts = Counter()

    for record in SeqIO.parse(fasta_file, "fasta"):
        sequence = str(record.seq).upper()
        if len(sequence) >= 5:
            start_4mer_counts[sequence[:4]] += 1
            for i in range(len(sequence) - 4):
                prefix = sequence[i:i + 4]
                suffix = sequence[i + 4]
                model[prefix][suffix] += 1

    # Convert counts to probabilities
    for prefix, suffixes in model.items():
        total = sum(suffixes.values())
        for suffix in suffixes:
            suffixes[suffix] /= total

    total_starts = sum(start_4mer_counts.values())
    start_4mer_probs = {k: v / total_starts for k, v in start_4mer_counts.items()}

    return model, start_4mer_probs


def save_model(model, start_4mer_probs, filename):
    """
    Saves the model and starting 4-mer probabilities to a file.

    Parameters:
    - model: The 5-mer model.
    - start_4mer_probs: Probabilities of starting 4-mers.
    - filename: The filename to save the model.
    """
    with open(filename, 'wb') as f:
        pickle.dump({'model': model, 'start_4mer_probs': start_4mer_probs}, f)


def main():
    fasta_file = "../../../data/training data/uniprot_sprot.fasta"
    output_filename = "../../../data/models/5mer_model.pkl"

    start_time = time.time()
    model, start_4mer_probs = build_5mer_model(fasta_file)
    end_time = time.time()

    save_model(model, start_4mer_probs, output_filename)

    print(f"Model saved to {output_filename}. Took {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
