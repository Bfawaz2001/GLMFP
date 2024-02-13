import pickle
from Bio import SeqIO
from collections import defaultdict
import os
import time

def build_trigram_model(fasta_file):
    """
    Build a trigram (3-mer) model from a given FASTA file of proteins.

    Args:
    fasta_file (str): Path to the FASTA file.

    Returns:
    dict: A dictionary representing the trigram model.
    """
    # Initialize a defaultdict to count trigrams. The default factory is another defaultdict(int).
    trigram_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    # Parse the FASTA file
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequence = str(record.seq).upper()
        # Generate trigrams and count them
        for i in range(len(sequence) - 2):
            first = sequence[i]
            second = sequence[i + 1]
            third = sequence[i + 2]
            trigram_counts[first][second][third] += 1

    # Convert counts to probabilities
    trigram_model = {}
    for first, second_dict in trigram_counts.items():
        trigram_model[first] = {}
        for second, third_dict in second_dict.items():
            total = sum(third_dict.values())
            trigram_model[first][second] = {third: count / total for third, count in third_dict.items()}

    return trigram_model

def save_model(model, filename):
    """
    Save the model to a file using pickle.

    Args:
    model (dict): The trigram model to save.
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
    dict: The loaded trigram model.
    """
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

def main():
    fasta_file = "../data/uniprot_sprot.fasta"
    output_filename = "../data/models/trigram_model.pkl"

    # Ensure the FASTA file exists
    if not os.path.exists(fasta_file):
        print(f"Error: The file {fasta_file} does not exist.")
        return

    # Start timing
    start_time = time.time()

    # Build the trigram model
    trigram_model = build_trigram_model(fasta_file)

    # Stop Timing
    end_time = time.time()

    # Save the model
    save_model(trigram_model, output_filename)

    print(f"Trigram model saved to {output_filename}")

    # Print the elapsed time
    print(f"Model creation and saving took {end_time - start_time} seconds.")

if __name__ == "__main__":
    main()
