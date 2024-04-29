import random
import textwrap
import time
import subprocess
import os
import json
import torch
import torch.nn as nn
import pickle
from Bio.SeqUtils import molecular_weight, IsoelectricPoint
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F
import xml.etree.ElementTree as ET
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# File path to the models
NGRAM_MODEL_PATH = "../../data/models/n-gram/"
NN_MODEL_PATH = '../../data/models/neural network/'
TRANS_MODEL_PATH = '../../data/models/transformer/'

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
    """Function to return a defaultdict with int as the default factory."""
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
    """LSTM model for protein sequence generation."""
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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=2500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # Shape [sequence length, 1, embedding size] for correct broadcasting
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x expected to be of shape [sequence length, batch size, embedding size]
        x = x + self.pe[:x.size(0), :]  # Corrected indexing for dynamic sequence lengths
        return self.dropout(x)


class TransformerProteinGenerator(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=3, dropout=0.1, batch_first=True):
        super(TransformerProteinGenerator, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_model * 2, dropout, batch_first=batch_first)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.encoder = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, vocab_size)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output


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
    """
    Load the Neural Network (LSTM) model and its associated label encoder from disk.

    Parameters:
    - model_path (str): The file path to the NN model's state dictionary.
    - encoder_path (str): The file path to the label encoder.

    Returns:
    - model: The loaded NN (LSTM) model.
    - label_encoder: The loaded label encoder.
    """
    model = LSTMProteinGenerator(vocab_size=26, embedding_dim=64, hidden_dim=128, num_layers=4)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    model.to("cpu")  # Ensure the model is fully on the CPU

    # Load the encoder as before
    with open(encoder_path, 'rb') as file:
        label_encoder = pickle.load(file)

    return model, label_encoder


def load_trans_model_and_encoder(model_path, encoder_path):
    """
    Load the Transformer model and its associated label encoder from disk.

    Parameters:
    - model_path (str): The file path to the Transformer model's state dictionary.
    - encoder_path (str): The file path to the label encoder.

    Returns:
    - model: The loaded Transformer model.
    - label_encoder: The loaded label encoder.
    """
    # Initialize the model structure as defined in the transformer.py
    # Adjust these parameters as necessary to match your model's configuration
    vocab_size = 26  # Update this based on your dataset
    model = TransformerProteinGenerator(vocab_size=vocab_size, d_model=64, nhead=4, num_layers=3, dropout=0.1)

    # Load the model's state dictionary
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    model.to("cpu")  # Ensure the model is fully on the CPU

    # Load the label encoder
    with open(encoder_path, 'rb') as file:
        label_encoder = pickle.load(file)

    return model, label_encoder


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
    all_aas = 'ABCDEFGHIKLMNOPQRSTUVWXYZ'
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


def parse_interproscan_results(data):
    """
    Parses the JSON output from InterProScan to extract detailed annotations for each protein.
    Includes the protein ID, signature descriptions, accessions, GO terms, and pathway information.

    Args:
    - data (dict): The JSON-decoded data from an InterProScan results file.

    Returns:
    - summary (dict): A dictionary summarizing the annotations for each protein.
    """
    summary = {}
    if 'results' in data:
        for result in data['results']:
            xrefs = result.get('xref', [])
            if not xrefs:
                continue
            protein_id = xrefs[0].get('id', "Unknown Protein ID")
            # Assuming 'sequence' is available at the 'result' level
            protein_sequence = result.get('sequence', '')

            annotations = []
            matches = result.get('matches', [])
            for match in matches:
                signature = match.get('signature', {})
                signature_description = signature.get('description', 'No description available')
                signature_accession = signature.get('accession', 'N/A')

                entry = signature.get('entry', {}) or {}
                go_terms = [{'id': go.get('id'), 'name': go.get('name'), 'database': go.get('db'),
                             'category': go.get('category')} for go in entry.get('goXRefs', [])]

                pathways = [{'name': pathway.get('name'), 'id': pathway.get('id'),
                             'database': pathway.get('databaseName')} for pathway in entry.get('pathwayXRefs', [])]

                match_details = []
                for location in match.get('locations', []):
                    start = location.get('start') - 1  # Assuming 0-based indexing for Python string slicing
                    end = location.get('end')
                    sequence_match = protein_sequence[start:end]  # Extract matched sequence slice
                    match_details.append({'start': start + 1, 'end': end, 'score': location.get('score', 'N/A'),
                                          'evalue': location.get('evalue', 'N/A'), 'sequence_match': sequence_match})

                annotations.append({'signature_accession': signature_accession, 'description': signature_description,
                                    'GO_terms': go_terms, 'pathways': pathways, 'match_details': match_details})

            summary[protein_id] = annotations
    return summary


def parse_blastp_xml(xml_file_path, summary_output_path):
    """
    Parse DIAMOND BLASTp XML output, calculate the match percentage for each hit,
    and write a formatted summary to a text file with a maximum of 120 characters per line.
    Additionally, print the total number of hits found. Now also includes gaps, query and hit sequences,
    alignment coverage, and a bit score.

    Args:
        xml_file_path (str): Path to the DIAMOND BLASTp XML output file.
        summary_output_path (str): Path to the output text file where the summary will be written.
    """
    # Parse the XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    total_hits = 0  # Initialize total hits counter

    with open(summary_output_path, 'w') as summary_file:
        for iteration in root.findall('.//Iteration'):
            query_id = iteration.find('Iteration_query-def').text
            hits = iteration.findall('.//Hit')
            num_hits = len(hits)
            total_hits += num_hits  # Update total hits counter

            summary_file.write(f"Query ID: {query_id}\n")
            summary_file.write(f"Number of hits: {num_hits}\n\n")

            for hit in hits:
                hit_id = hit.find('Hit_id').text
                hit_def = hit.find('Hit_def').text
                hit_accession = hit.find('Hit_accession').text
                hsp = hit.find('.//Hsp')
                hsp_bit_score = hsp.find('Hsp_bit-score').text
                hsp_evalue = hsp.find('Hsp_evalue').text
                hsp_score = hsp.find('Hsp_score').text
                hsp_identity = int(hsp.find('Hsp_identity').text)
                hsp_align_len = int(hsp.find('Hsp_align-len').text)
                hsp_gaps = hsp.find('Hsp_gaps').text
                hsp_qseq = hsp.find('Hsp_qseq').text
                hsp_hseq = hsp.find('Hsp_hseq').text
                match_percentage = (hsp_identity / hsp_align_len) * 100 if hsp_align_len > 0 else 0
                coverage = (hsp_align_len / int(iteration.find('Iteration_query-len').text)) * 100

                # Writing using textwrap to ensure each line does not exceed 120 characters
                hit_info = (f"  Hit ID: {hit_id}, Hit Def: {hit_def}, Accession: {hit_accession}, "
                            f"E-value: {hsp_evalue}, Score: {hsp_score}, Bit Score: {hsp_bit_score}, "
                            f"Identity: {hsp_identity}/{hsp_align_len} ({match_percentage:.2f}%), "
                            f"Coverage: {coverage:.2f}%, Gaps: {hsp_gaps}, "
                            f"Query Seq: {hsp_qseq}, Hit Seq: {hsp_hseq}\n")
                wrapped_hit_info = textwrap.fill(hit_info, width=120, subsequent_indent='    ')
                summary_file.write(wrapped_hit_info + "\n\n")

    # After processing all queries, print the total number of hits found
    print(f"Total number of hits found: {total_hits}")


def write_interpro_summary(summary, output_file):
    with open(output_file, 'w') as file:
        for protein_id, annotations in summary.items():
            file.write(f"Protein ID: {protein_id}\n")
            if not annotations:  # If there are no annotations for this protein
                file.write("  No matches found.\n\n")
                continue  # Skip to the next protein

            for annotation in annotations:
                # Ensuring values are not None before writing
                signature_accession = annotation['signature_accession'] or "N/A"
                description = annotation['description'] or "No description available"
                file.write(f"  Signature Accession: {signature_accession}\n")
                file.write(f"  Description: {textwrap.fill(description, width=120, subsequent_indent='    ')}\n")

                if annotation['GO_terms']:
                    go_terms_str = ", ".join(
                        [f"{go.get('name', 'N/A')} ({go.get('id', 'N/A')})" for go in annotation['GO_terms']])
                    file.write(f"  GO Terms: {textwrap.fill(go_terms_str, width=120, subsequent_indent='    ')}\n")

                if annotation['pathways']:
                    pathways_str = ", ".join(
                        [f"{pathway.get('name', 'N/A')} ({pathway.get('id', 'N/A')})" for pathway in
                         annotation['pathways']])
                    file.write(f"  Pathways: {textwrap.fill(pathways_str, width=120, subsequent_indent='    ')}\n")

                if annotation['match_details']:
                    match_details_str = "; ".join([f"Start: {md.get('start', 'N/A')}, End: {md.get('end', 'N/A')}, "
                                                   f"Score: {md.get('score', 'N/A')}, Evalue: {md.get('evalue', 'N/A')}"
                                                   f", "
                                                   f"Matched Sequence: {md.get('sequence_match', 'N/A')}"
                                                   for md in annotation['match_details']])
                    file.write(
                        f"  Match Details: {textwrap.fill(match_details_str, width=120, subsequent_indent='    ')}\n")

                file.write("\n")


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
            protein = generate_complex_protein(model, encoder, min_length, max_length)
            formatted_protein = '\n'.join(protein[j:j + 60] for j in range(0, len(protein), 60))
            file.write(">Protein_{}\n{}\n".format(i + 1, formatted_protein))

    print("Generated proteins saved to {}".format(output_filename))


def generate_proteins_trans_interface(model, model_type, encoder):
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
            protein = generate_complex_protein(model, encoder, min_length, max_length)
            formatted_protein = '\n'.join(protein[j:j + 60] for j in range(0, len(protein), 60))
            file.write(">Protein_{}\n{}\n".format(i + 1, formatted_protein))

    print("Generated proteins saved to {}".format(output_filename))


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
            next_aa = random.choices(list(model['2mer_model'][current_aa].keys()),
                                     weights=model['2mer_model'][current_aa].values())[0]
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
            next_aa_options = model['3mer_model'].get(current_pair, {})
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
            next_aa_options = model['5mer_model'].get(current_4mer, {})
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
            next_aa_options = model['6mer_model'].get(current_5mer, {})
            if next_aa_options:
                next_aa = random.choices(list(next_aa_options.keys()), weights=next_aa_options.values())[0]
                protein.append(next_aa)
                current_5mer = ''.join(protein[-5:])
            else:
                break  # If no valid continuation is found

    return ''.join(protein)  # Convert the list of amino acids back into a string and return it.


def generate_complex_protein(model, label_encoder, min_length, max_length, temperature=1.5):
    """
        Generate a protein sequence using the Neural Network or Transformer model, starting with a seed sequence.

        Parameters:
        - model: The loaded model either Neural Network LSTM or Transformer.
        - label_encoder: The loaded label encoder.
        - min_length (int): Minimum length of the generated protein sequence.
        - max_length (int): Maximum length of the generated protein sequence.

        Returns:
        - The generated protein sequence as a string.
        """
    # Initialize with the start sequence 'M' (Methionine)
    start_seq = 'M'
    device = next(model.parameters()).device  # Identify if the model is on CPU or GPU
    sequence = [label_encoder.transform([start_seq])[0]]  # Encode the start sequence
    generated_sequence = start_seq  # Start the sequence with Methionine
    target_length = random.randint(min_length, max_length)  # Determine the target sequence length

    with torch.no_grad():
        while len(generated_sequence) < target_length:
            input_tensor = torch.tensor([sequence[-50:]], dtype=torch.long).to(device)  # Prepare the input tensor
            output = model(input_tensor)  # Obtain logits from the model
            output = output[:, -1, :] / temperature  # Focus on the last output predictions
            probabilities = F.softmax(output, dim=1)  # Apply softmax to convert logits to probabilities
            next_index = torch.multinomial(probabilities, 1).item()
            # Sample from the probability distribution

            next_aa = label_encoder.inverse_transform([next_index])[0]  # Decode the predicted index back to amino acid
            generated_sequence += next_aa  # Append the predicted amino acid to the generated sequence
            sequence.append(next_index)  # Update the sequence with the predicted index for next iteration

    return generated_sequence


def run_diamond_blastp(fasta_file, diamond_db_path, database):
    """
    Executes a DIAMOND BLASTP search for a given FASTA file against a specified protein database
    and reports on the results.

    This function runs the DIAMOND BLASTP tool to compare protein sequences contained within a FASTA file against a
    pre-built DIAMOND database (either 'nr' or 'swissprot'). It generates an output file containing the search results,
    checks for the presence of matches, and calculates the average identity percentage of those matches. If no matches
    are found, it notifies the user that no file was created or that the file is empty.

    Args:
        fasta_file (str): Path to the FASTA file containing protein sequences for comparison.
        diamond_db_path (str): Path to the pre-built DIAMOND database to be used for the search.
        database (str): Name of the database ('nr' or 'swissprot') against which the search is performed, used for
                        reporting purposes.

    Returns:
        None.

    Raises:
        subprocess.CalledProcessError: If the DIAMOND BLASTP command fails during execution, this error is raised with
                                        details about the failure.
    """
    results_filename = "{}_{}_diamond_blastp.xml".format(database, fasta_file.removesuffix('.fasta'))
    results_directory = os.path.join(DIAMOND_RESULTS_PATH, fasta_file.removesuffix('.fasta'))
    os.makedirs(results_directory, exist_ok=True)
    output_file = os.path.join(results_directory, results_filename)

    diamond_cmd = [
        'diamond', 'blastp',
        '--db', diamond_db_path,
        '--query', GENERATED_PROTEINS_RESULTS_PATH+fasta_file,
        '--out', output_file,
        '--outfmt', '5',
        '--max-target-seqs', '5',
        '--evalue', '1e-3',
    ]

    try:
        print("\nRunning DIAMOND BLASTP against the {} database for {}...".format(database.upper(), fasta_file))
        start_time = time.time()
        subprocess.run(diamond_cmd, check=True)
        end_time = time.time()

        # Check if output file is empty or does not exist
        if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
            print("\nNo matches found. No file created or the file is empty for {}.".format(fasta_file))
            return  # Exit the function as there's nothing further to do

        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            print("\nAnalysis complete. Results are saved in {}, "
                  "and took {:.2f} seconds.".format(output_file, end_time - start_time))
            # Define a path for the summary file
            summary_output_path = os.path.join(results_directory,
                                               f"{database}_{fasta_file.removesuffix('.fasta')}_summary.txt")
            # Call the function to parse XML and write summary
            parse_blastp_xml(output_file, summary_output_path)
            print(f"DIAMOND BLASTp summary written to {summary_output_path}")
        else:
            print("\nNo matches found. No file created or the file is empty for {}.".format(fasta_file))

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
        '--outformat', 'json',
        '--outfile', output_file,
    ]

    # Execute the command
    print("\nRUNNING InterProScan...")
    start_time = time.time()
    result = subprocess.run(command, capture_output=True, text=True)
    end_time = time.time()

    output_file = output_file + '.json.json'

    # Check if the file with the extra .json extension exists
    if os.path.exists(output_file):
        # Construct the correct output filename by removing one .json extension
        correct_output_file = output_file.removesuffix('.json')

        # Rename the file to remove the extra .json extension
        os.rename(output_file, correct_output_file)
        # Reassign the output_file to thr correct name to be used to create the summary file
        output_file = correct_output_file

    # Check if the subprocess executed successfully
    if result.returncode != 0:
        print("Error running InterProScan:")
        print(result.stderr)
        return

    # Verify output file existence
    if not os.path.exists(output_file):
        print("Expected output file not found: {}".format(output_file))
        return

    print("InterProScan results saved to {},".format(output_file),
          "and took {:.2f} seconds.".format(end_time - start_time))

    # Assuming parse_interproscan_results and write_annotations_to_csv are implemented correctly
    try:
        with open(output_file) as json_file:
            data = json.load(json_file)
        summary = parse_interproscan_results(data)
        csv_output_file = os.path.join(results_directory, "{}_summary.txt".format(selected_file.removesuffix('.fasta')))
        write_interpro_summary(summary, csv_output_file)
        print("Annotations summary saved to {}".format(csv_output_file))

    except Exception as e:
        print("An error occurred while parsing or writing the summary: {}".format(e))


def summary_protein_sequences(selected_file):
    sequences = list(parse_fasta(GENERATED_PROTEINS_RESULTS_PATH + selected_file))
    results_directory = os.path.join(ANALYSIS_SUMMARY_PATH, selected_file.removesuffix('.fasta'))
    graphs_directory = os.path.join(results_directory, "aa_composition_graphs/")
    os.makedirs(results_directory, exist_ok=True)
    os.makedirs(graphs_directory, exist_ok=True)

    total_compositions = Counter()

    summary_path = os.path.join(results_directory, "{}_summary.txt".format(selected_file.removesuffix('.fasta')))
    with open(summary_path, 'w') as summary_file:
        for i, (header, sequence) in enumerate(sequences, 1):
            composition = Counter(sequence)
            total_compositions += composition  # Update total composition

            # Calculate percentages for the current protein
            total_aa_count = sum(composition.values())
            percentage_composition = {aa: (count / total_aa_count) * 100 for aa, count in composition.items()}

            # Write details to summary file
            summary_file.write("Protein {}\n".format(i))
            summary_file.write("Amino Acid Counts and Frequencies:\n")
            for aa, count in composition.items():
                frequency = percentage_composition[aa]
                summary_file.write("  {}: Count = {}, Frequency = {:.2f}%\n".format(aa, count, frequency))

            entropy = calculate_shannon_entropy(sequence)
            mw, ip = calculate_physicochemical_properties(sequence)

            summary_file.write("Shannon Entropy: {:.4f}\n".format(entropy))
            summary_file.write("Molecular Weight: {:.2f}\n".format(mw))
            summary_file.write("Isoelectric Point: {:.2f}\n\n".format(ip))

            # Plot and save individual protein composition graph
            individual_graph_path = os.path.join(graphs_directory, "protein_{}_composition.png".format(i))
            plot_aa_composition(percentage_composition, 'Protein {} Amino Acid Composition'.format(i),
                                individual_graph_path)

        # Calculate overall composition percentages for total graph
        total_aa_count = sum(total_compositions.values())
        overall_percentage_composition = {aa: (count / total_aa_count) * 100 for aa, count in
                                          total_compositions.items()}

        # Write overall statistics to the summary file
        summary_file.write("\nOverall Amino Acid Counts and Frequencies:\n")
        for aa, count in total_compositions.items():
            frequency = overall_percentage_composition[aa]
            summary_file.write(f"  {aa}: Count = {count}, Frequency = {frequency:.2f}%\n")

        # Plot and save total composition graph
        total_graph_path = os.path.join(graphs_directory, "total_composition.png")
        plot_aa_composition(overall_percentage_composition, 'Overall Amino Acid Composition', total_graph_path)

        print("\nSummary file saved to {}".format(summary_path))
        print("Amino Acid composition graphs saved to {}.".format(graphs_directory))


def diamond_blastp_menu(selected_file):
    """
    Presents analysis tool options for the selected FASTA file and executes the chosen analysis.

    Args:
    selected_file (str): The filename of the selected FASTA file for analysis.

    The function supports comparing the selected file against the NCBI nr database, as well as the
    training data.
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
    print("3. Summary of proteins sequences (amino acid composition, shannon entropy, physical chemistry)")
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


def run_nn_model():
    model_type = 'nn_lstm'
    model_path = NN_MODEL_PATH + 'nn_model.pt'
    encoder_path = NN_MODEL_PATH + 'nn_label_encoder.pkl'

    model, encoder = load_nn_model_and_encoder(model_path, encoder_path)
    generate_proteins_nn_interface(model, model_type, encoder)


def run_trans_model():
    model_type = 'transformer'
    model_path = TRANS_MODEL_PATH + 'transformer_model.pt'
    encoder_path = TRANS_MODEL_PATH + 'transformer_label_encoder.pkl'

    model, encoder = load_trans_model_and_encoder(model_path, encoder_path)
    generate_proteins_trans_interface(model, model_type, encoder)


def ngram_menu():
    """
    Display the model selection menu and handle user input.
    """
    print("\nSelect an N-gram model:")
    print("1. 2-mer")
    print("2. 3-mer")
    print("3. 5-mer")
    print("4. 6-mer")
    print("5. Back to Main Menu")
    choice = input("Enter your choice: ")

    if choice == '1':
        model_type = '2mer'
        selected_model = NGRAM_MODEL_PATH + '2mer_model.pkl'
    elif choice == '2':
        model_type = '3mer'
        selected_model = NGRAM_MODEL_PATH + '3mer_model.pkl'
    elif choice == '3':
        model_type = '5mer'
        selected_model = NGRAM_MODEL_PATH + '5mer_model.pkl'
    elif choice == '4':
        model_type = '6mer'
        selected_model = NGRAM_MODEL_PATH + '6mer_model.pkl'
    elif choice == '5':
        print("\nGoing back to Main Menu...")
        return
    else:
        print("Invalid choice. Returning to main menu.")
        return

    model = load_ngram_model(selected_model)
    generate_proteins_ngram_interface(model, model_type)


def model_menu():
    """
    Display the model selection menu and handle user input.
    """
    print("\nSelect a model:")
    print("1. N-Gram Models")
    print("2. Neural Network (LSTM) Model")
    print("3. Transformer Model")
    print("4. Back to Main Menu")
    choice = input("Enter your choice: ")

    if choice == '1':
        ngram_menu()
    elif choice == '2':
        run_nn_model()
    elif choice == '3':
        run_trans_model()
    elif choice == '4':
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
            print("\nExiting the program...")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main_menu()

