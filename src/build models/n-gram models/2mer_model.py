import os
import pickle
import time
from collections import defaultdict, Counter
from Bio import SeqIO


def build_bigram_model(fasta_file):
    """
    Builds a bigram model from a FASTA file containing protein sequences.
    Parameters:
    - fasta_file: Path to the FASTA file.
    Returns:
    - bigram_model: A dictionary where keys are amino acids and values are dictionaries
      of subsequent amino acids with their transition probabilities.
    - start_amino_acid_counts: A Counter object counting the frequency of each starting amino acid.
    """
    bigram_counts = defaultdict(lambda: defaultdict(int))
    start_amino_acid_counts = Counter()

    for record in SeqIO.parse(fasta_file, "fasta"):
        sequence = str(record.seq).upper()
        start_amino_acid_counts[sequence[0]] += 1
        for i in range(len(sequence) - 1):
            first = sequence[i]
            second = sequence[i + 1]
            bigram_counts[first][second] += 1

    bigram_model = {first: {second: count / sum(inner_dict.values())
                            for second, count in inner_dict.items()}
                    for first, inner_dict in bigram_counts.items()}

    total_starts = sum(start_amino_acid_counts.values())
    start_amino_acid_probs = {aa: count / total_starts for aa, count in start_amino_acid_counts.items()}

    return bigram_model, start_amino_acid_counts, start_amino_acid_probs


def save_model(model, start_amino_acid_counts, start_amino_acid_probs, filename):
    """
    Saves the bigram model, starting amino acid counts, and their probabilities to a file.
    Parameters:
    - model: The bigram model to be saved.
    - start_amino_acid_counts: Starting amino acid counts to be saved.
    - start_amino_acid_probs: Starting amino acid probabilities to be saved.
    - filename: Path to the file where the model and counts will be saved.
    """
    model_data = {'bigram_model': model, 'start_amino_acids': dict(start_amino_acid_counts), 'start_amino_acid_probs': start_amino_acid_probs}
    with open(filename, 'wb') as file:
        pickle.dump(model_data, file)


def main():
    """
    Main function to build and save the bigram model from protein sequences.
    """
    fasta_file = "../../../data/training data/uniprot_sprot.fasta"
    output_filename = "../../../data/models/n-gram/2mer_model.pkl"

    if not os.path.exists(fasta_file):
        print("Error: The file {} does not exist.".format(fasta_file))
        return

    start_time = time.time()
    bigram_model, start_amino_acid_counts, start_amino_acid_probs = build_bigram_model(fasta_file)
    end_time = time.time()

    save_model(bigram_model, start_amino_acid_counts, start_amino_acid_probs, output_filename)

    print("Model saved to {}. Took {:.2f} seconds.".format(output_filename, end_time - start_time))


if __name__ == "__main__":
    main()
