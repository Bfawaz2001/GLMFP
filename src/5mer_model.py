import pickle
from Bio import SeqIO
from collections import defaultdict
import os
import time

def default_dict():
    return defaultdict(int)

def build_5mer_model(fasta_file):
    """
    Build a 5-mer model from a given FASTA file of proteins.
    """
    model = defaultdict(default_dict)
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequence = str(record.seq).upper()
        for i in range(len(sequence) - 4):
            five_mer = sequence[i:i + 5]
            prefix = five_mer[:-1]
            suffix = five_mer[-1]
            model[prefix][suffix] += 1

    # Convert counts to probabilities
    for prefix, suffix_dict in model.items():
        total = float(sum(suffix_dict.values()))
        for suffix in suffix_dict:
            suffix_dict[suffix] /= total

    return model


def save_model(model, filename):
    """
    Save the model to a file using pickle.
    """
    with open(filename, 'wb') as f:
        # Convert defaultdict to a regular dict for pickling
        regular_dict_model = {prefix: dict(suffixes) for prefix, suffixes in model.items()}
        pickle.dump(regular_dict_model, f)


def main():
    fasta_file = "/Users/bfawa/Desktop/GLMFP_Test1/data/uniprot_sprot.fasta"
    output_filename = "../data/models/5mer_model.pkl"

    # Ensure the FASTA file exists
    if not os.path.exists(fasta_file):
        print("File does not exist.")
        return

    # Start timing
    start_time = time.time()

    model = build_5mer_model(fasta_file)

    # Stop Timing
    end_time = time.time()

    save_model(model, output_filename)

    print(f"Model saved to {output_filename}")

    # Print the elapsed time
    print(f"Model creation and saving took {end_time - start_time} seconds.")

if __name__ == "__main__":
    main()