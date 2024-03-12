import time
import matplotlib.pyplot as plt
from collections import defaultdict
import json


def generate_dummy_sequences(num_sequences=100000, sequence_length=250):
    """Generate dummy sequences for demonstration purposes."""
    import random
    sequences = []
    for _ in range(num_sequences):
        sequence = ''.join(random.choices('ACDEFGHIKLMNPQRSTVWY', k=sequence_length))
        sequences.append(sequence)
    return sequences


def build_ngram_model(sequences, n):
    """Builds an n-gram model from the given sequences and calculates probabilities."""
    ngram_counts = defaultdict(int)
    total_ngrams = 0

    # Count each n-gram occurrence
    for sequence in sequences:
        for i in range(len(sequence) - n + 1):
            ngram = sequence[i:i + n]
            ngram_counts[ngram] += 1
            total_ngrams += 1

    # Calculate probabilities
    ngram_probabilities = {ngram: count / total_ngrams for ngram, count in ngram_counts.items()}

    # Save probabilities to file
    with open(f'ngram_probabilities_{n}.json', 'w') as file:
        json.dump(ngram_probabilities, file, indent=4)

    return ngram_counts  # Return counts if needed for further processing


def main():
    sequences = generate_dummy_sequences()
    ns = range(2, 31)  # From 2-mer to 30-mer
    times = []

    for n in ns:
        start_time = time.time()
        build_ngram_model(sequences, n)
        end_time = time.time()
        times.append(end_time - start_time)

    plt.plot(ns, times, marker='o')
    plt.title('Time to Build N-gram Models (2-mer to 30-mer)')
    plt.xlabel('N-mer')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()