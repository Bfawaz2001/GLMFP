import pickle
from collections import defaultdict, Counter
from Bio import SeqIO
import time

def defaultdict_int():
    """Create a defaultdict with int as the default factory."""
    return defaultdict(int)

def build_7mer_model(fasta_file):
    """
    Builds a 7-mer model from protein sequences in a FASTA file.

    Args:
        fasta_file (str): Path to the FASTA file containing protein sequences.

    Returns:
        tuple: A tuple containing the 7-mer model and start 6-mer probabilities.
    """
    model = defaultdict(defaultdict_int)
    start_6mer_counts = Counter()

    for record in SeqIO.parse(fasta_file, "fasta"):
        sequence = str(record.seq).upper()
        if len(sequence) >= 7:
            # Increment start 6-mer count
            start_6mer_counts[sequence[:6]] += 1
            # Update 7-mer counts
            for i in range(len(sequence) - 6):
                prefix = sequence[i:i + 6]
                suffix = sequence[i + 6]
                model[prefix][suffix] += 1

    # Normalize counts to probabilities
    for prefix, suffixes in model.items():
        total = sum(suffixes.values())
        for suffix in suffixes:
            suffixes[suffix] = suffixes[suffix]/total

    total_starts = sum(start_6mer_counts.values())
    start_6mer_probs = {k: v / total_starts for k, v in start_6mer_counts.items()}

    return model, start_6mer_probs

def save_model(model, start_6mer_probs, filename):
    """
    Saves the 7-mer model and start 6-mer probabilities to a file using pickle.

    Args:
        model (dict): The 7-mer model.
        start_6mer_probs (dict): The start 6-mer probabilities.
        filename (str): Filename for saving the model.
    """
    with open(filename, 'wb') as f:
        pickle.dump({'model': model, 'start_6mer_probs': start_6mer_probs}, f)

def main():
    """
    Main execution function to build and save the 7-mer model.
    """
    fasta_file = "../../../data/training data/uniprot_sprot.fasta"  # Adjust with actual FASTA file path
    output_filename = "../../../data/models/7mer_model.pkl"  # Adjust with desired output path

    start_time = time.time()
    model, start_6mer_probs = build_7mer_model(fasta_file)
    end_time = time.time()

    save_model(model, start_6mer_probs, output_filename)
    print(f"Model saved to {output_filename}. Took {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()

