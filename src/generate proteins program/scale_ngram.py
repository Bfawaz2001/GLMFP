from Bio import SeqIO
import time
import matplotlib.pyplot as plt


def generate_ngrams(sequence, n):
    ngrams = [sequence[i:i + n] for i in range(len(sequence) - n + 1)]
    return ngrams


def calculate_probabilities(ngrams):
    total_ngrams = len(ngrams)
    probabilities = {}
    for ngram in set(ngrams):
        probabilities[ngram] = ngrams.count(ngram) / total_ngrams
    return probabilities


def main(fasta_file):
    sequences = [str(record.seq) for record in SeqIO.parse(fasta_file, "fasta")]

    times = []
    ns = range(2, 31)

    for n in ns:
        start_time = time.time()

        # Generate n-grams and calculate probabilities for each sequence
        for sequence in sequences:
            ngrams = generate_ngrams(sequence, n)
            probabilities = calculate_probabilities(ngrams)

        end_time = time.time()
        times.append(end_time - start_time)
        print(f"Time to generate and calculate probabilities for {n}-mers: {end_time - start_time} seconds")

    plt.figure(figsize=(10, 6))
    plt.plot(ns, times, marker='o', linestyle='-', color='b')
    plt.title("Scaling of n-gram Model Creation Time")
    plt.xlabel("n-gram")
    plt.ylabel("Time (seconds)")
    plt.xticks(ns)
    plt.grid(True)

    # Save the figure
    plt.savefig("/ngram_scaling_graph.png", dpi=300)  # Saves the figure to a file
    plt.close()  # Close the plot to prevent it from displaying in the notebook

    print("The scaling graph has been saved to ngram_scaling_graph.png")


if __name__ == "__main__":
    fasta_file = "../../results/generated proteins/3mer_test2.fasta"  # Change this to the path of your FASTA file
    main(fasta_file)
