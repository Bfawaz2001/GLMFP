import random
import time
from collections import defaultdict
import subprocess
import os
import pandas as pd
import json
import csv
import torch
import torch.nn as nn
import pickle
from Bio.SeqUtils import molecular_weight, IsoelectricPoint
from collections import Counter
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F

# File path to the models
NGRAM_MODEL_PATH = "../../data/models/n-gram/"
RNN_MODEL_PATH = '../../data/models/neural network/'

# File Paths to results directories
GENERATED_PROTEINS_RESULTS_PATH = "../../results/generated proteins/"
INTERPRO_RESULTS_PATH = "../../results/interpro results/"
ANALYSIS_SUMMARY_PATH = "../../results/analysis summary/"
DIAMOND_RESULTS_PATH = "../../results/diamond blastp results/"

# Diamond Database paths for DIAMOND BLASTp
DIAMOND_NR_DB_PATH = "../../data/diamond db/nr.dmnd"
DIAMOND_SwissProt_DB_PATH = "../../data/diamond db/uniprot_sprot.dmnd"

# InterProScan script path and user email address
IPRSCAN5_PATH = "../../data/interpro script/iprscan5.py"
EMAIL = "b.fawaz2001@gmail.com"


def defaultdict_int():
    """Returns a defaultdict with int as the default factory, replacing lambda."""
    return defaultdict(int)


def parse_fasta(file_path):
    """Parses a FASTA file and yields sequence ID and sequence pairs."""
    with open(file_path, 'r') as fasta_file:
        seq_id = None
        sequence = []
        for line in fasta_file:
            line = line.strip()
            if line.startswith(">"):
                if seq_id:  # Yield the previous sequence before starting a new one
                    yield seq_id, ''.join(sequence)
                seq_id = line[1:]  # Capture the sequence ID immediately after '>'
                sequence = []  # Reset sequence for the next entry
            else:
                sequence.append(line)
        if seq_id:  # Ensure the last sequence in the file is also yielded
            yield seq_id, ''.join(sequence)


class LSTMProteinGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(LSTMProteinGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        out = self.fc(lstm_out)
        return out


def calculate_amino_acid_composition(sequences):
    """Calculate amino acid composition for a list of sequences as percentages."""
    all_sequences_composition = Counter()
    for _, sequence in sequences:
        all_sequences_composition += Counter(sequence)
    total_aa_count = sum(all_sequences_composition.values())
    # Calculate percentage composition
    percentage_composition = {aa: (count / total_aa_count) * 100 for aa, count in all_sequences_composition.items()}
    return percentage_composition


def plot_aa_composition(composition, title, save_path):
    """Plot amino acid composition in alphabetical order with percentages."""
    # Ensure all 20 amino acids are represented in the plot, even if they are not in the composition
    all_aas = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    composition_full = {aa: composition.get(aa, 0) for aa in all_aas}

    labels, values = zip(*sorted(composition_full.items()))  # Sort by amino acid
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color='skyblue')
    plt.xlabel('Amino Acid')
    plt.ylabel('Percentage (%)')
    plt.title(title)
    plt.xticks(rotation=45)  # Improve label readability
    plt.savefig(save_path)
    plt.close()


def calculate_shannon_entropy(sequence):
    frequency = Counter(sequence)
    entropy = -sum((freq / len(sequence)) * math.log2(freq / len(sequence)) for freq in frequency.values())
    return entropy

def clean_sequence(sequence):
    """Remove 'X' in the protein sequence."""
    cleaned_sequence = sequence.replace('X', '')
    return cleaned_sequence


def calculate_physicochemical_properties(sequence):
    cleaned_sequence = clean_sequence(sequence)  # Clean the sequence first
    mw = molecular_weight(cleaned_sequence, seq_type='protein')
    ip = IsoelectricPoint.IsoelectricPoint(cleaned_sequence).pi()
    return mw, ip


def parse_interproscan_results(json_file):
    with open(json_file) as f:
        data = json.load(f)

    summary = {}
    for result in data['results']:
        protein_id = result['sequence']['id']
        annotations = []
        for entry in result.get('entries', []):
            for annotation in entry.get('goAnnotations', []):
                annotations.append({
                    'domain': entry['entry']['name'],
                    'description': entry['entry']['description'],
                    'GO': annotation['identifier']
                })
        summary[protein_id] = annotations
    return summary


def write_annotations_to_csv(summary, output_file):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Protein ID', 'Domain', 'Description', 'GO'])

        for protein_id, annotations in summary.items():
            for annotation in annotations:
                writer.writerow([
                    protein_id,
                    annotation['domain'],
                    annotation['description'],
                    annotation['GO']
                ])


def load_ngram_model(filename):
    """
    Load a model from a file using pickle.

    Args:
    filename (str): Path to the file where the model is saved.

    Returns:
    dict: The loaded model.
    """
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model


def load_nn_model_and_encoder(model_path, encoder_path):
    model = LSTMProteinGenerator(vocab_size=27, embedding_dim=64, hidden_dim=128, num_layers=4)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to("cpu")

    with open(encoder_path, 'rb') as file:
        label_encoder = pickle.load(file)

    return model, label_encoder


def generate_nn_protein(model, label_encoder, min_length, max_length, temperature=1):
    start_seq = 'M'
    device = next(model.parameters()).device
    sequence = [label_encoder.transform([start_seq])[0]]
    generated_sequence = start_seq
    target_length = random.randint(min_length, max_length)

    while len(generated_sequence) < target_length:
        current_input = torch.tensor([sequence[-50:]], dtype=torch.long).to(device)
        with torch.no_grad():
            output = model(current_input)
            output = output[:, -1, :] / temperature  # Scale the logits before applying softmax
            probabilities = F.softmax(output, dim=1)
            next_index = torch.multinomial(probabilities, 1).item()

        next_aa = label_encoder.inverse_transform([next_index])[0]
        generated_sequence += next_aa
        sequence.append(next_index)

    return generated_sequence


def generate_ngram_protein(model, min_length, max_length, model_type):
    """
    Generate a single protein sequence using the specified model.

    Args:
        model (dict): The loaded model, which can be 2mer, 3mer, 5mer, or 6mer.
        min_length (int): The minimum length of the generated protein sequence.
        max_length (int): The maximum length of the generated protein sequence.
        model_type (str): The type of model used for generation ('2mer', '3mer', '5mer', '6mer').

    Returns:
        str: A generated protein sequence as a string.
    """
    length = random.randint(min_length, max_length)  # Determine the length of the protein sequence.
    protein = []  # Initialize an empty list to store the sequence.

    if model_type == '2mer':
        # Generate protein sequence based on 2mer model.
        start_aas = list(model['start_amino_acid_probs'].keys())  # Get starting amino acids.
        start_probs = list(model['start_amino_acid_probs'].values())  # Get probabilities of starting amino acids.
        start_aa = random.choices(start_aas, weights=start_probs)[0]  # Choose starting amino acid.
        protein.append(start_aa)  # Add the starting amino acid to the sequence.
        current_aa = start_aa

        # Continue adding amino acids based on the 2mer model probabilities until the desired length is reached.
        while len(protein) < length:
            next_aa = random.choices(list(model['bigram_model'][current_aa].keys()),
                                     weights=model['bigram_model'][current_aa].values())[0]
            protein.append(next_aa)
            current_aa = next_aa

    elif model_type == '3mer':
        # Similar approach for 3mer, starting with a pair of amino acids and expanding the sequence.
        start_pairs = list(model['start_2mer_probs'].keys())
        start_probs = list(model['start_2mer_probs'].values())
        start_pair = random.choices(start_pairs, weights=start_probs)[0]
        protein.extend(start_pair)  # Add both amino acids of the starting pair to the sequence.
        current_pair = start_pair

        # Generate the rest of the protein sequence based on 3mer model probabilities.
        while len(protein) < length:
            next_aa_options = model['trigram_model'].get(current_pair, {})
            if next_aa_options:
                next_aa = random.choices(list(next_aa_options.keys()), weights=next_aa_options.values())[0]
                protein.append(next_aa)
                current_pair = protein[-2] + next_aa  # Update the current pair for the next iteration.
            else:
                break  # Exit the loop if no valid next amino acid is found.

    elif model_type == '5mer':
        # Use probabilities to select the starting 4-mer
        start_4mers = list(model['start_4mer_probs'].keys())
        start_4mer_probs = list(model['start_4mer_probs'].values())
        start_4mer = random.choices(start_4mers, weights=start_4mer_probs)[0]
        for aa in start_4mer:  # Add each amino acid of the 4-mer to the protein sequence
            protein.append(aa)
        current_4mer = start_4mer

        while len(protein) < length:
            next_aa_options = model['model'].get(current_4mer, {})
            if next_aa_options:
                next_aa = random.choices(list(next_aa_options.keys()), weights=next_aa_options.values())[0]
                protein.append(next_aa)
                current_4mer = ''.join(
                    protein[-4:])  # Update the current 4-mer based on the last 4 amino acids in the sequence
            else:
                break  # Stop if no valid continuation is found

    elif model_type == '6mer':
        start_5mers = list(model['start_5mer_probs'].keys())
        start_5mer_probs = list(model['start_5mer_probs'].values())
        start_5mer = random.choices(start_5mers, weights=start_5mer_probs)[0]
        protein.extend(list(start_5mer))
        current_5mer = start_5mer

        while len(protein) < length:
            next_aa_options = model['model'].get(current_5mer, {})
            if next_aa_options:
                next_aa = random.choices(list(next_aa_options.keys()), weights=next_aa_options.values())[0]
                protein.append(next_aa)
                current_5mer = ''.join(protein[-5:])
            else:
                break  # If no valid continuation is found

    elif model_type == '10mer':
        start_9mers = list(model['start_9mer_probs'].keys())
        start_9mer_probs = list(model['start_9mer_probs'].values())
        start_9mer = random.choices(start_9mers, weights=start_9mer_probs)[0]
        protein.extend(list(start_9mer))
        current_5mer = start_9mer

        while len(protein) < length:
            next_aa_options = model['model'].get(current_5mer, {})
            if next_aa_options:
                next_aa = random.choices(list(next_aa_options.keys()), weights=next_aa_options.values())[0]
                protein.append(next_aa)
                current_5mer = ''.join(protein[-9:])
            else:
                break  # If no valid continuation is found

    return ''.join(protein)  # Convert the list of amino acids back into a string and return it.


def generate_proteins_nn_interface(model, model_type, encoder):
    """
    Interface for generating proteins using the selected model.
    """
    base_filename = input("\nEnter the name for the generated proteins file (without extension, "
                          "model name added automatically to beginning): ")

    output_filename = "{}_{}.fasta".format(model_type, base_filename)

    num_proteins = int(input("Enter the number of proteins to be created: "))
    min_length = int(input("Enter the minimum length of the amino acids: "))
    max_length = int(input("Enter the maximum length of the amino acids: "))

    with open(GENERATED_PROTEINS_RESULTS_PATH + output_filename, 'w') as file:
        for i in range(num_proteins):
            protein = generate_nn_protein(model, encoder, min_length, max_length)
            formatted_protein = '\n'.join(protein[j:j + 60] for j in range(0, len(protein), 60))
            file.write(">Protein_{}\n{}\n".format(i + 1, formatted_protein))

    print("Generated proteins saved to {}".format(output_filename))


def generate_proteins_ngram_interface(model, model_type):
    """
    Interface for generating proteins using the selected model.
    """
    base_filename = input("\nEnter the name for the generated proteins file (without extension, "
                          "model name added automatically to beginning): ")

    output_filename = "{}_{}.fasta".format(model_type, base_filename)

    num_proteins = int(input("Enter the number of proteins to be created: "))
    min_length = int(input("Enter the minimum length of the amino acids: "))
    max_length = int(input("Enter the maximum length of the amino acids: "))

    with open(GENERATED_PROTEINS_RESULTS_PATH + output_filename, 'w') as file:
        for i in range(num_proteins):
            protein = generate_ngram_protein(model, min_length, max_length, model_type)
            formatted_protein = '\n'.join(protein[j:j + 60] for j in range(0, len(protein), 60))
            file.write(">Protein_{}\n{}\n".format(i + 1, formatted_protein))

    print("Generated proteins saved to {}".format(output_filename))


def add_headers_to_diamond_output(output_file, headers):
    # Define your headers as a list of column names. For example:
    # headers = ["qseqid", "sseqid", "pident", "length", "mismatch", "gapopen", "qstart", "qend", "sstart", "send", "evalue", "bitscore"]

    # Read the TSV file without headers
    df = pd.read_csv(output_file, sep='\t', header=None)

    # Assign the headers to the dataframe
    df.columns = headers

    # Write the dataframe back to the file with headers
    df.to_csv(output_file, sep='\t', index=False)

def run_diamond_blastp(fasta_file, diamond_db_path, database):
    """
    This function utilizes DIAMOND BLASTP to compare a selected FASTA file against a specified protein database
    and reports the percentage of identity matches. It is designed to handle both 'nr' and 'swissprot' databases.

    Args:
        fasta_file (str): The FASTA file for comparison.
        diamond_db_path (str): Path to the DIAMOND database.
        database (str): The database name ('nr' or 'swissprot') for contextual output.
    """
    results_filename = f"{database}_{fasta_file.removesuffix('.fasta')}_diamond_blastp.tsv"
    output_file = f"{DIAMOND_RESULTS_PATH}{results_filename}"

    diamond_cmd = [
        'diamond', 'blastp',
        '--db', diamond_db_path,
        '--query', f"{GENERATED_PROTEINS_RESULTS_PATH}{fasta_file}",
        '--out', output_file,
        '--outfmt', '6',
        '--max-target-seqs', '1',
        '--evalue', '1e-3'
    ]

    try:
        print(f"\nRunning DIAMOND BLASTP against the {database.upper()} database for {fasta_file}...")
        subprocess.run(diamond_cmd, check=True)
        print(f"\nAnalysis complete. Results are saved in {output_file}.")

        headers = ["qseqid", "sseqid", "pident", "length", "mismatch", "gapopen", "qstart", "qend", "sstart", "send",
                   "evalue",
                   "bitscore"]
        add_headers_to_diamond_output(output_file, headers)

        # Process and report results
        df = pd.read_csv(output_file, sep='\t')
        avg_identity = df['pident'].mean()
        print(f"Average identity: {avg_identity:.2f}%")
    except subprocess.CalledProcessError as e:
        print("Error during DIAMOND BLASTP execution: ", e)


def run_interpro_scan(selected_file, email):
    """
    Run InterProScan on a given FASTA file by invoking the iprscan5.py script with JSON output.

    Parameters:
    - selected_file: Name of the FASTA file to be analyzed.
    - email: Email address for job submission.
    """
    # Ensuring paths are absolute for reliability
    fasta_file_path = os.path.join(GENERATED_PROTEINS_RESULTS_PATH, selected_file)
    results_directory = os.path.join(INTERPRO_RESULTS_PATH, selected_file.removesuffix('.fasta'))
    os.makedirs(results_directory, exist_ok=True)
    output_file = os.path.join(results_directory, selected_file.removesuffix('.fasta'))

    # Basic command to run iprscan5.py with required arguments
    command = [
        'python', IPRSCAN5_PATH,
        '--email', email,
        '--sequence', fasta_file_path,
        '--stype', 'p',
        '--outfile', output_file,
    ]

    # Execute the command
    start_time = time.time()
    result = subprocess.run(command, capture_output=True, text=True)
    end_time = time.time()

    # Check if the subprocess executed successfully
    if result.returncode != 0:
        print("Error running InterProScan:")
        print(result.stderr)
        return

    # Verify output file existence
    if not os.path.exists(output_file):
        print(f"Expected output file not found: {output_file}")
        return

    print(f"InterProScan results saved to {output_file}, "
          f"and took {end_time - start_time:.2f} seconds.")

    # Assuming parse_interproscan_results and write_annotations_to_csv are implemented correctly
    try:
        summary = parse_interproscan_results(output_file)
        csv_output_file = os.path.join(results_directory, f"{selected_file.removesuffix('.fasta')}_summary.csv")
        write_annotations_to_csv(summary, csv_output_file)
        print("Annotations summary saved to {}".format(csv_output_file))
    except Exception as e:
        print("An error occurred while parsing or writing the summary: {}".format(e))


def diamond_blastp_menu(selected_file):
    """
    Presents analysis tool options for the selected FASTA file and executes the chosen analysis.

    Args:
    selected_file (str): The filename of the selected FASTA file for analysis.

    The function supports comparing the selected file against the NCBI nr database,
    labeling functionalities (InterProScan), and visualizing proteins (AlphaFold).
    Future implementations can replace 'pass' with actual function calls.
    """
    print("\nDIAMOND BLASTp Menu")
    print("1. Compare against NCBI nr (non-redundant proteins) database")
    print("2. Compare against Reviewed SwissProt proteins (training data) database")
    print("3. Go back to Main Menu")
    option = input("Select an option for analysis: ")

    if option == '1':
        database = 'nr'
        run_diamond_blastp(selected_file, DIAMOND_NR_DB_PATH, database)
    elif option == '2':
        database = 'swissprot'
        run_diamond_blastp(selected_file, DIAMOND_SwissProt_DB_PATH, database)
    elif option == "3":
        print("\nGoing Back to main menu...")
    else:
        print("\nInvalid option. Returning to main menu.")
        return


def summary_protein_sequences(selected_file):
    sequences = list(parse_fasta(GENERATED_PROTEINS_RESULTS_PATH + selected_file))

    # Create specific directories
    results_directory = os.path.join(ANALYSIS_SUMMARY_PATH, selected_file.removesuffix('.fasta'))
    graphs_directory = os.path.join(results_directory, "aa composition graphs/")
    os.makedirs(results_directory, exist_ok=True)
    os.makedirs(graphs_directory, exist_ok=True)

    all_compositions = defaultdict(int)

    # Prepare a single summary file instead of individual files for each sequence
    summary_path = os.path.join(results_directory, f"{selected_file.removesuffix('.fasta')}_summary.txt")
    with open(summary_path, 'w') as summary_file:
        for i, (header, sequence) in enumerate(sequences, 1):
            composition = Counter(sequence)
            for aa, freq in composition.items():
                all_compositions[aa] += freq
            entropy = calculate_shannon_entropy(sequence)
            mw, ip = calculate_physicochemical_properties(sequence)

            # Write detailed summary to the single file
            summary_file.write(
                f"Protein {i}\n"
                f"Shannon Entropy: {entropy:.4f}\n"
                f"Molecular Weight: {mw:.2f}\n"
                f"Isoelectric Point: {ip:.2f}\n\n")

            # Plot and save AA composition graph
            graph_path = os.path.join(graphs_directory, "protein_{}_composition.png".format(i))
            plot_aa_composition(composition, 'Amino Acid Composition of Protein {}'.format(i), graph_path)

    # Plot and save overall AA composition graph
    total_graph_path = os.path.join(graphs_directory, "total_composition.png")
    plot_aa_composition(all_compositions, 'Overall Amino Acid Composition', total_graph_path)

    print("\nSummary file saved to {}".format(summary_path))
    print("Amino Acid composition Graphs saved to {}.".format(graphs_directory))


def analysis_options(selected_file):
    """
    Presents analysis tool options for the selected FASTA file and executes the chosen analysis.

    Args:
    selected_file (str): The filename of the selected FASTA file for analysis.

    The function supports comparing the selected file against the NCBI nr database,
    labeling functionalities (InterProScan), and visualizing proteins (AlphaFold).
    Future implementations can replace 'pass' with actual function calls.
    """
    print("\nAnalysis Tool Selection Menu")
    print("1. Compare against known protein sequences (DIAMOND BLASTp)")
    print("2. Label Protein Functionalities (InterProScan)")
    print("3. Summary of proteins sequences (amino acid composition, shannon entropy, physiochemistry)")
    print("4. Re-select protein FASTA file")
    print("5. Go back to Main Menu")
    option = input("Select an option for analysis: ")

    if option == '1':
        diamond_blastp_menu(selected_file)
    elif option == '2':
        run_interpro_scan(selected_file, EMAIL)
    elif option == "3":
        summary_protein_sequences(selected_file)
    elif option == "4":
        analyse_proteins_menu()
    elif option == "5":
        print("\nGoing Back to main menu...")
        return
    else:
        print("\nInvalid option. Returning to main menu.")
        return

    analysis_options(selected_file)


def analyse_proteins_menu():
    """
    Display a menu for protein analysis options.

    Lists available FASTA files from the generated proteins directory and prompts the user
    to select one for further analysis. Validates the user's selection and proceeds
    to analysis options.
    """
    try:
        files = [f for f in os.listdir(GENERATED_PROTEINS_RESULTS_PATH) if f.endswith('.fasta')]
        if not files:
            print("No FASTA files found in the generated proteins directory.")
            return

        print("\nSelect a FASTA file to analyse:")
        for i, file in enumerate(files, 1):
            print("{}. {}".format(i, file))

        file_selection = int(input("Enter the number of the file: ")) - 1

        # Validate user selection
        if file_selection < 0 or file_selection >= len(files):
            print("Invalid selection. Please enter a valid number.")
            return

        selected_file = files[file_selection]
        analysis_options(selected_file)

    except Exception as e:
        print("An error occurred: {}".format(e))


def ngram_model_menu():
    """
    Display the model selection menu and handle user input.
    """
    print("\nSelect an N-gram model:")
    print("1. 2-mer")
    print("2. 3-mer")
    print("3. 5-mer")
    print("4. 6-mer")
    print("5. 10-mer")
    print("6. Back to Main Menu")
    choice = input("Enter your choice: ")

    if choice == '1':
        model_type = '2mer'
        filename = NGRAM_MODEL_PATH + '2mer_model.pkl'
    elif choice == '2':
        model_type = '3mer'
        filename = NGRAM_MODEL_PATH + '3mer_model.pkl'
    elif choice == '3':
        model_type = '5mer'
        filename = NGRAM_MODEL_PATH + '5mer_model.pkl'
    elif choice == '4':  # Assuming '4' is the new option for 6-mer
        model_type = '6mer'
        filename = NGRAM_MODEL_PATH + '6mer_model.pkl'
    elif choice == '5':
        model_type = '10mer'
        filename = NGRAM_MODEL_PATH + '10mer_model.pkl'
    elif choice == '6':
        print("\nGoing back to Main Menu...")
        return
    else:
        print("Invalid choice. Returning to main menu.")
        return

    model = load_ngram_model(filename)
    generate_proteins_ngram_interface(model, model_type)


def rnn_model_menu():
    print("\nSelect a Neural Network model:")
    print("1. RNN (LSTM)")
    print("2. Back to Main Menu")
    choice = input("Enter your choice: ")

    if choice == '1':
        model_type = 'nn_lstm_pytorch'
        model_path = RNN_MODEL_PATH + 'nn_pytorch.pt'
        encoder_path = RNN_MODEL_PATH + 'nn_label_encoder.pkl'
    elif choice == '2':
        print("\nGoing back to Main Menu...")
        return
    else:
        print("Invalid choice. Returning to main menu.")
        return

    model, encoder = load_nn_model_and_encoder(model_path, encoder_path)
    generate_proteins_nn_interface(model, model_type, encoder)


def model_menu():
    """
    Display the model selection menu and handle user input.
    """
    print("\nSelect a model:")
    print("1. N-gram Models")
    print("2. RNN Models")
    print("3. Back to Main Menu")
    choice = input("Enter your choice: ")

    if choice == '1':
        ngram_model_menu()
    elif choice == '2':
        rnn_model_menu()
    elif choice == '3':
        print("\nGoing back to Main Menu...")
        return
    else:
        print("Invalid choice. Returning to main menu.")
        return


def main_menu():
    """
    Display the main menu and handle user input for program navigation.

    This function presents the user with options to use a model, analyse proteins,
    or quit the program. It ensures that the user input is valid and calls the
    appropriate function based on the user's choice.
    """
    while True:
        print("\nMain Menu:")
        print("1. Use a model")
        print("2. Analyse Generated Proteins")
        print("3. Quit")
        choice = input("Enter your choice: ")

        if choice == '1':
            model_menu()
        elif choice == '2':
            analyse_proteins_menu()
        elif choice == '3':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main_menu()
