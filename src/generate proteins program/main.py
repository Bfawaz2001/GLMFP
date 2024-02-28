import json
import pickle
import random
import time
from collections import defaultdict
import subprocess
import os
import pandas as pd
import requests

#File path to the models
MODEL_PATH = "../../data/models/"
RESULTS_PATH = "../../data/generated proteins/"
DIAMOND_RESULTS_PATH = "../../data/diamond blastp results/"
DIAMOND_DB_PATH = "../../data/diamond db/uniprot_sprot.dmnd"
INTERPRO_RESULTS_PATH = "../../data/interpro results/"
ALPHA_FOLD_RESULTS_PATH = "../../data/alpha fold results/"

def defaultdict_int():
    """Returns a defaultdict with int as the default factory, replacing lambda."""
    return defaultdict(int)

def load_model(filename):
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

def load_rnn_model(model_path):
    return

def generate_protein_rnn(model, min_length, max_length):
    return
# Implementation for generating a protein sequence with the RNN model
# This will involve initializing a sequence and iteratively predicting the next amino acid until
# the sequence reaches the desired length


def generate_protein(model, min_length, max_length, model_type):
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
                current_4mer = ''.join(protein[-4:])  # Update the current 4-mer based on the last 4 amino acids in the sequence
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

    return ''.join(protein)  # Convert the list of amino acids back into a string and return it.


def compare_against_ncbi_nr(fasta_file):
    """
    Compares a selected FASTA file against the NCBI nr (non-redundant) database using DIAMOND BLASTP and reports
    the percentage of matches.

    Args:
        fasta_file (str): The name of the FASTA file selected for comparison.

    Enhancements include optimized DIAMOND BLASTP execution for large databases and more informative output regarding
    the percentage of sequence matches.
    """
    results_filename = input("Please enter the name of the generated proteins file: ").strip()
    if not results_filename:
        print("Invalid filename. Please provide a valid name.")
        return
    output_file = DIAMOND_RESULTS_PATH + results_filename  # Ensure correct output path

    diamond_cmd = [
        'diamond', 'blastp',
        '--db', DIAMOND_DB_PATH,
        '--query', RESULTS_PATH+fasta_file,
        '--out', output_file,
        '--outfmt', '6',
        '--max-target-seqs', '10',
        '--evalue', '0.001',
    ]

    try:
        print(f"\nRunning DIAMOND BLASTP against NCBI NR database for {fasta_file}...")
        start_time = time.time()
        subprocess.run(diamond_cmd, check=True)
        end_time = time.time()
        print(f"\nAnalysis complete. Results are saved in {output_file}.")
        print(f"Time taken: {start_time-end_time} seconds")

        # Further code to parse the output file and calculate the percentage of matches
        df = pd.read_csv(output_file, sep='\t', header=None,
                         names=['qseqid', 'sseqid', 'pident', 'length', 'mismatch', 'gapopen', 'qstart', 'sstart',
                                'evalue', 'bitscore'])
        avg_pident = df['pident'].mean()
        print(f"Average percentage of identity: {avg_pident}%")

    except subprocess.CalledProcessError as e:
        print("\nError during DIAMOND execution: ", e)

def label_functionalities(selected_file):
    """
    Query the InterPro API to retrieve functional annotations for a given protein sequence in a FASTA file.
    Enhanced to provide more accessible and informative output.
    """
    fasta_file_path = f'{RESULTS_PATH}{selected_file}'
    output_file_path = f'{INTERPRO_RESULTS_PATH}{selected_file}_interpro_results.json'

    try:
        # Read the FASTA file content
        with open(fasta_file_path, 'r') as file:
            fasta_content = file.read()

        # API URL for demonstration purposes, replace with actual query to InterPro API
        url = 'https://www.ebi.ac.uk/interpro/api/entry/interpro/protein/uniprot/P04637'

        # Send the GET request to the InterPro API
        response = requests.get(url)

        if response.status_code == 200:
            results = response.json()  # Assuming the response is in JSON format

            # Enhance the output structure for readability
            enhanced_output = {
                "message": "InterProScan analysis completed successfully.",
                "results_path": output_file_path,
                "details": results  # This inserts the actual generated proteins from the API into our output
            }

            # Save the enhanced output to a file in JSON format
            with open(output_file_path, 'w') as output_file:
                json.dump(enhanced_output, output_file, indent=4)

            print(f"InterProScan analysis completed. Results saved to {output_file_path}.")

        else:
            print(f"An error occurred while running InterProScan: {response.status_code}")

    except Exception as e:
        print(f"An error occurred: {e}")

def visualise_proteins(selected_file):
    """
    Placeholder function for visualizing proteins with AlphaFold.
    Future implementation should include the actual call to AlphaFold and handle the visualization.

    Args:
    selected_file (str): The filename of the selected FASTA file for protein visualization.
    """
    fasta_file_path = f'{RESULTS_PATH}{selected_file}'
    output_directory = ALPHA_FOLD_RESULTS_PATH + selected_file.split('.')[0] + '_alpha_fold_results/'

    # Make sure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # AlphaFold command setup
    # Note: This command assumes you have a script or executable named 'run_alphafold' that is set up to
    # run AlphaFold with the necessary parameters and environment configurations.
    alpha_fold_cmd = [
        'run_alphafold',  # This should be the path to your AlphaFold running script or executable
        '--fasta_path', fasta_file_path,
        '--output_dir', output_directory,
        '--model_names', 'model_1,model_2,model_3,model_4,model_5',
        '--preset', 'full_dbs',
        '--use_gpu'
    ]

    try:
        print(f"Running AlphaFold for {selected_file}...")
        subprocess.run(alpha_fold_cmd, check=True)
        print(f"Protein visualization completed. Check the output in {output_directory}.")
    except subprocess.CalledProcessError as e:
        print(f"Error running AlphaFold: {e}")

def generate_proteins_interface(model, model_type):
    """
    Interface for generating proteins using the selected model.
    """
    base_filename = input("\nEnter the name for the generated proteins file (without extension): ")
    output_filename = f"{base_filename}.fasta"

    num_proteins = int(input("Enter the number of proteins to be created: "))
    min_length = int(input("Enter the minimum length of the amino acids: "))
    max_length = int(input("Enter the maximum length of the amino acids: "))

    with open(RESULTS_PATH+output_filename, 'w') as file:
        for i in range(num_proteins):
            protein = generate_protein(model, min_length, max_length, model_type)
            formatted_protein = '\n'.join(protein[j:j + 60] for j in range(0, len(protein), 60))
            file.write(f">Protein_{i + 1}\n{formatted_protein}\n")

    print(f"Generated proteins saved to {output_filename}")

def analyse_options(selected_file):
    """
    Presents analysis tool options for the selected FASTA file and executes the chosen analysis.

    Args:
    selected_file (str): The filename of the selected FASTA file for analysis.

    The function supports comparing the selected file against the NCBI nr database,
    labeling functionalities (InterProScan), and visualizing proteins (AlphaFold).
    Future implementations can replace 'pass' with actual function calls.
    """
    print("\nAnalysis Tool Selection Menu")
    print("1. Compare against NCBI nr (non-redundant) database")
    print("2. Label the functionalities (InterProScan)")
    print("3. Visualise Proteins (AlphaFold)")
    print("4. Go back to Main Menu")
    option = input("Select an option for analysis: ")

    if option == '1':
        compare_against_ncbi_nr(selected_file)
    elif option == '2':
        label_functionalities(selected_file)
    elif option == '3':
        visualise_proteins(selected_file)
    elif option == "4":
        print("\nGoing Back to main menu...")
    else:
        print("\nInvalid option. Returning to main menu.")
        return

def analyse_proteins_menu():
    """
    Display a menu for protein analysis options.

    Lists available FASTA files from the generated proteins directory and prompts the user
    to select one for further analysis. Validates the user's selection and proceeds
    to analysis options.
    """
    try:
        files = [f for f in os.listdir(RESULTS_PATH) if f.endswith('.fasta')]
        if not files:
            print("No FASTA files found in the generated proteins directory.")
            return

        print("\nSelect a FASTA file to analyse:")
        for i, file in enumerate(files, 1):
            print(f"{i}. {file}")

        file_selection = int(input("Enter the number of the file: ")) - 1

        # Validate user selection
        if file_selection < 0 or file_selection >= len(files):
            print("Invalid selection. Please enter a valid number.")
            return

        selected_file = files[file_selection]
        analyse_options(selected_file)

    except Exception as e:
        print(f"An error occurred: {e}")

def ngram_model_menu():
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
        filename = MODEL_PATH + '2mer_model.pkl'
    elif choice == '2':
        model_type = '3mer'
        filename = MODEL_PATH + '3mer_model.pkl'
    elif choice == '3':
        model_type = '5mer'
        filename = MODEL_PATH + '5mer_model.pkl'
    elif choice == '4':  # Assuming '4' is the new option for 6-mer
        model_type = '6mer'
        filename = MODEL_PATH + '6mer_model.pkl'
    elif choice == '5':
        print("\nGoing back to Main Menu...")
        return
    else:
        print("Invalid choice. Returning to main menu.")
        return

    model = load_model(filename)
    generate_proteins_interface(model, model_type)

def rnn_model_menu():
    pass


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
