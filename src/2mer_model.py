import pickle
from Bio import SeqIO
from collections import defaultdict
import os
import time


def build_bigram_model(fasta_file):
    """
    Build a bigram (2-mer) model from a given FASTA file of proteins.

    Args:
    fasta_file (str): Path to the FASTA file.

    Returns:
    dict: A dictionary representing the bigram model.
    """
    # Initialize a defaultdict to count bigrams. The default factory is set to int, starting from 0.
    bigram_counts = defaultdict(lambda: defaultdict(int))

    # Parse the FASTA file
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequence = str(record.seq).upper()
        # Generate bigrams and count them
        for i in range(len(sequence) - 1):
            first = sequence[i]
            second = sequence[i + 1]
            bigram_counts[first][second] += 1

    # Convert counts to probabilities
    bigram_model = {}
    for first, inner_dict in bigram_counts.items():
        total = sum(inner_dict.values())
        bigram_model[first] = {second: count / total for second, count in inner_dict.items()}

    return bigram_model


def save_model(model, filename):
    """
    Save the model to a file using pickle.

    Args:
    model (dict): The bigram model to save.
    filename (str): Path to the file where the model should be saved.
    """
    with open(filename, 'wb') as file:
        pickle.dump(model, file)


def load_model(filename):
    """
    Load a model from a file using pickle.

    Args:
    filename (str): Path to the file where the model is saved.

    Returns:
    dict: The loaded bigram model.
    """
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model


def main():
    fasta_file = "../data/uniprot_sprot.fasta"
    output_filename = "../data/models/2mer_model.pkl"

    # Ensure the FASTA file exists
    if not os.path.exists(fasta_file):
        print(f"Error: The file {fasta_file} does not exist.")
        return

    # Start timing
    start_time = time.time()

    # Build the bigram model
    bigram_model = build_bigram_model(fasta_file)

    # Stop Timing
    end_time = time.time()

    # Save the model
    save_model(bigram_model, output_filename)

    print(f"Bigram model saved to {output_filename}")

    # Print the elapsed time
    print(f"Model creation and saving took {end_time - start_time} seconds.")


if __name__ == "__main__":
    main()
