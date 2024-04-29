import pickle
from collections import defaultdict, Counter
from Bio import SeqIO
import time


def defaultdict_int():
    """Function to return a defaultdict with int as the default factory."""
    return defaultdict(int)


def build_6mer_model(fasta_file):
    """
    Builds a 6-mer model from a FASTA file.

    Args:
        fasta_file (str): Path to the FASTA file containing protein sequences.

    Returns:
        A tuple containing the 6-mer model and start 5-mer probabilities.
    """
    model = defaultdict(defaultdict_int)
    start_5mer_counts = Counter()

    for record in SeqIO.parse(fasta_file, "fasta"):
        sequence = str(record.seq).upper()
        if len(sequence) >= 6:
            start_5mer_counts[sequence[:5]] += 1
            for i in range(len(sequence) - 5):
                prefix = sequence[i:i + 5]
                suffix = sequence[i + 5]
                model[prefix][suffix] += 1

    # Convert counts to probabilities
    for prefix, suffixes in model.items():
        total = sum(suffixes.values())
        for suffix in suffixes:
            suffixes[suffix] = suffixes[suffix] / total

    total_starts = sum(start_5mer_counts.values())
    start_5mer_probs = {k: v / total_starts for k, v in start_5mer_counts.items()}

    return model, start_5mer_probs


def save_model(model, start_5mer_probs, filename):
    """
    Saves the 6-mer model and start 5-mer probabilities to a file.

    Args:
        model (dict): The 6-mer model.
        start_5mer_probs (dict): The start 5-mer probabilities.
        filename (str): The path where to save the model.
    """
    regular_dict_model = {prefix: dict(suffixes) for prefix, suffixes in model.items()}
    model_data = {'6mer_model': regular_dict_model, 'start_5mer_probs': dict(start_5mer_probs)}
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)


def main():
    """
    Main function to build and save the 6-mer model.
    """
    fasta_file = "../../../data/training data/uniprot_sprot.fasta"  # Update with your FASTA file path
    output_filename = "../../../data/models/n-gram/6mer_model.pkl"  # Update with your output path

    start_time = time.time()
    model, start_5mer_probs = build_6mer_model(fasta_file)
    end_time = time.time()

    save_model(model, start_5mer_probs, output_filename)
    print("Model saved to {}. Took {:.2f} seconds.".format(output_filename, end_time - start_time))


if __name__ == "__main__":
    main()
    
