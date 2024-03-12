import pickle
from collections import defaultdict, Counter
from Bio import SeqIO
import time


def defaultdict_int():
    """Function to create a defaultdict with int as the default factory."""
    return defaultdict(int)


def build_10mer_model(fasta_file, probability_threshold=0.001):
    """
    Builds a 10-mer model from a given FASTA file.

    Args:
        fasta_file (str): Path to the FASTA file.

    Returns:
        tuple: A tuple containing the 10-mer model and start 9-mer probabilities.
    """
    model = defaultdict(defaultdict_int)
    start_9mer_counts = Counter()

    for record in SeqIO.parse(fasta_file, "fasta"):
        sequence = str(record.seq).upper()
        if len(sequence) >= 10:
            # Count the start 9-mer
            start_9mer_counts[sequence[:9]] += 1
            # Count each 10-mer and its subsequent amino acid
            for i in range(len(sequence) - 9):
                prefix = sequence[i:i + 9]
                suffix = sequence[i + 9]
                model[prefix][suffix] += 1

        # Apply thresholding
    for prefix, suffixes in model.items():
        total = sum(suffixes.values())
        for suffix in list(suffixes):
            probability = suffixes[suffix] / total
            if probability < probability_threshold:
                del suffixes[suffix]  # Remove low-probability events
            else:
                suffixes[suffix] = probability  # Store as probability directly

    total_starts = sum(start_9mer_counts.values())
    start_9mer_probs = {k: v / total_starts for k, v in start_9mer_counts.items()}

    return model, start_9mer_probs

def save_model(model, start_9mer_probs, filename):
    """
    Saves the model and starting 9-mer probabilities to a file using pickle.

    Args:
        model (dict): The 10-mer model.
        start_9mer_probs (dict): The starting 9-mer probabilities.
        filename (str): The filename where the model should be saved.
    """
    with open(filename, 'wb') as f:
        pickle.dump({'model': model, 'start_9mer_probs': start_9mer_probs}, f)


def main():
    """
    Main function to build and save the 10-mer model from a FASTA file.
    """
    fasta_file = "../../../data/training data/uniprot_sprot.fasta"  # Update this path
    output_filename = "../../../data/models/n-gram/10mer_model.pkl"  # Update this path

    start_time = time.time()
    model, start_9mer_probs = build_10mer_model(fasta_file)
    end_time = time.time()

    save_model(model, start_9mer_probs, output_filename)

    print(f"Model saved to {output_filename}. Took {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()