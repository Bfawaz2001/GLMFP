from collections import Counter
from Bio import SeqIO
from Bio.SeqUtils import molecular_weight
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import math

def calculate_shannon_entropy(sequence):
    frequency = Counter(sequence)
    entropy = -sum((freq / len(sequence)) * math.log2(freq / len(sequence)) for freq in frequency.values())
    return entropy

def clean_sequence(sequence):
    """Remove 'X', 'Z', and 'B' in the protein sequence."""
    cleaned_sequence = sequence.replace('X', '').replace('Z', '').replace('B', '')
    return cleaned_sequence

def calculate_physicochemical_properties(sequence):
    cleaned_sequence = clean_sequence(sequence)  # Clean the sequence first
    mw = molecular_weight(cleaned_sequence, seq_type='protein')
    ip = ProteinAnalysis(cleaned_sequence).isoelectric_point()
    return mw, ip

def main(fasta_file):
    total_sequences = 0
    sum_lengths = 0
    sum_molecular_weight = 0
    sum_isoelectric_point = 0
    sum_entropy = 0
    max_length = 0
    min_length = float('inf')
    max_entropy = float('-inf')
    min_entropy = float('inf')
    max_molecular_weight = float('-inf')
    min_molecular_weight = float('inf')
    max_isoelectric_point = float('-inf')
    min_isoelectric_point = float('inf')

    for record in SeqIO.parse(fasta_file, "fasta"):
        sequence = str(record.seq)
        mw, ip = calculate_physicochemical_properties(sequence)
        entropy = calculate_shannon_entropy(sequence)
        length = len(clean_sequence(sequence))

        # Update sums and counts
        sum_lengths += length
        sum_molecular_weight += mw
        sum_isoelectric_point += ip
        sum_entropy += entropy
        total_sequences += 1

        # Update max and min values
        max_length = max(max_length, length)
        min_length = min(min_length, length)
        max_entropy = max(max_entropy, entropy)
        min_entropy = min(min_entropy, entropy)
        max_molecular_weight = max(max_molecular_weight, mw)
        min_molecular_weight = min(min_molecular_weight, mw)
        max_isoelectric_point = max(max_isoelectric_point, ip)
        min_isoelectric_point = min(min_isoelectric_point, ip)

    if total_sequences > 0:
        print("Average Molecular Weight:", sum_molecular_weight / total_sequences)
        print("Average Isoelectric Point:", sum_isoelectric_point / total_sequences)
        print("Average Shannon Entropy:", sum_entropy / total_sequences)
        print("Average Sequence Length:", sum_lengths / total_sequences)
        print("Maximum Amino Acid Sequence Length:", max_length)
        print("Minimum Amino Acid Sequence Length:", min_length)
        print("Maximum Shannon Entropy:", max_entropy)
        print("Minimum Shannon Entropy:", min_entropy)
        print("Maximum Molecular Weight:", max_molecular_weight)
        print("Minimum Molecular Weight:", min_molecular_weight)
        print("Maximum Isoelectric Point:", max_isoelectric_point)
        print("Minimum Isoelectric Point:", min_isoelectric_point)
        print("Total Number of Sequences:", total_sequences)
    else:
        print("No valid protein sequences found.")


if __name__ == "__main__":
    main('../../data/training data/uniprot_sprot.fasta')
