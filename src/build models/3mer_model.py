import os
import pickle
import time
from collections import defaultdict, Counter
from Bio import SeqIO


def build_trigram_model(fasta_file):
    """
    Builds a trigram model from a FASTA file containing protein sequences.

    Parameters:
    - fasta_file: Path to the FASTA file.

    Returns:
    - trigram_model: A dictionary where keys are pairs of amino acids and values
      are dictionaries of subsequent amino acids with their transition probabilities.
    - start_2mer_counts: A Counter object counting the frequency of each starting 2-mer.
    - start_2mer_probs: A dictionary with the probability of each starting 2-mer.
    """
    trigram_counts = defaultdict(lambda: defaultdict(int))
    start_2mer_counts = Counter()

    for record in SeqIO.parse(fasta_file, "fasta"):
        sequence = str(record.seq).upper()
        if len(sequence) >= 2:
            start_2mer_counts[sequence[:2]] += 1
            for i in range(len(sequence) - 2):
                first_2mer = sequence[i:i + 2]
                third = sequence[i + 2]
                trigram_counts[first_2mer][third] += 1

    trigram_model = {first_2mer: {third: count / sum(inner_dict.values())
                                  for third, count in inner_dict.items()}
                     for first_2mer, inner_dict in trigram_counts.items()}

    total_starts = sum(start_2mer_counts.values())
    start_2mer_probs = {start_2mer: count / total_starts for start_2mer, count in start_2mer_counts.items()}

    return trigram_model, start_2mer_counts, start_2mer_probs


def save_model(model, start_2mer_counts, start_2mer_probs, filename):
    """
    Saves the trigram model, starting 2-mer counts, and their probabilities to a file.

    Parameters:
    - model: The trigram model to be saved.
    - start_2mer_counts: Starting 2-mer counts to be saved.
    - start_2mer_probs: Starting 2-mer probabilities to be saved.
    - filename: Path to the file where the model and counts will be saved.
    """
    model_data = {'trigram_model': model, 'start_2mers': dict(start_2mer_counts), 'start_2mer_probs': start_2mer_probs}
    with open(filename, 'wb') as file:
        pickle.dump(model_data, file)


def main():
    """
    Main function to build and save the trigram model from protein sequences.
    """
    fasta_file = "../../data/uniprot_sprot.fasta"
    output_filename = "../../data/models/3mer_model.pkl"

    if not os.path.exists(fasta_file):
        print(f"Error: The file {fasta_file} does not exist.")
        return

    start_time = time.time()
    trigram_model, start_2mer_counts, start_2mer_probs = build_trigram_model(fasta_file)
    end_time = time.time()

    save_model(trigram_model, start_2mer_counts, start_2mer_probs, output_filename)

    print(f"Model saved to {output_filename}. Took {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
