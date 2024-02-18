import pickle
import random
from collections import defaultdict

#File path to the models
MODEL_PATH = "../../data/models/"
RESULTS_PATH = "../../data/results/"
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

def generate_protein(model, min_length, max_length, model_type):
    """
    Generate a single protein sequence using the specified model.

    Args:
    model (dict): The loaded model (either 2mer or 3mer).
    min_length (int): Minimum length of the protein.
    max_length (int): Maximum length of the protein.
    model_type (str): Type of the model ('2mer' or '3mer').

    Returns:
    str: A generated protein sequence.
    """
    length = random.randint(min_length, max_length)
    protein = []

    # 2mer generate proteins
    if model_type == '2mer':
        # Use probabilities to select the starting amino acid
        start_aas = list(model['start_amino_acid_probs'].keys())
        start_probs = list(model['start_amino_acid_probs'].values())
        start_aa = random.choices(start_aas, weights=start_probs)[0]
        protein.append(start_aa)
        current_aa = start_aa

        while len(protein) < length:
            next_aa = random.choices(list(model['bigram_model'][current_aa].keys()), weights=model['bigram_model'][current_aa].values())[0]
            protein.append(next_aa)
            current_aa = next_aa


    elif model_type == '3mer':
        # Use probabilities to select the starting 2-mer
        start_pairs = list(model['start_2mer_probs'].keys())
        start_probs = list(model['start_2mer_probs'].values())
        start_pair = random.choices(start_pairs, weights=start_probs)[0]
        protein.append(start_pair[0])
        protein.append(start_pair[1])
        current_pair = start_pair

        while len(protein) < length:
            next_aa_options = model['trigram_model'].get(current_pair, {})
            if next_aa_options:
                next_aa = random.choices(list(next_aa_options.keys()), weights=next_aa_options.values())[0]
                protein.append(next_aa)
                current_pair = protein[-2] + next_aa
            else:
                break  # Stop if no valid continuation is found

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

    return ''.join(protein)

def main_menu():
    """
    Display the main menu and handle user input.
    """
    while True:
        print("\nMain Menu:")
        print("1. Use a model")
        print("2. Quit")
        choice = input("Enter your choice (1 or 2): ")

        if choice == '1':
            model_menu()
        elif choice == '2':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")

def model_menu():
    """
    Display the model selection menu and handle user input.
    """
    print("\nSelect a model:")
    print("1. 2-mer")
    print("2. 3-mer")
    print("3. 5-mer")
    print("4. 6-mer")
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
    else:
        print("Invalid choice. Returning to main menu.")
        return

    model = load_model(filename)
    generate_proteins_interface(model, model_type)

def generate_proteins_interface(model, model_type):
    """
    Interface for generating proteins using the selected model.
    """
    base_filename = input("\nEnter the name for the generated proteins file (without extension): ")
    output_filename = f"{base_filename}.fasta"

    num_proteins = int(input("Enter the number of proteins to be created: "))
    min_length = int(input("Enter the minimum length of the proteins: "))
    max_length = int(input("Enter the maximum length of the proteins: "))

    with open(RESULTS_PATH+output_filename, 'w') as file:
        for i in range(num_proteins):
            protein = generate_protein(model, min_length, max_length, model_type)
            formatted_protein = '\n'.join(protein[j:j + 60] for j in range(0, len(protein), 60))
            file.write(f">Protein_{i + 1}\n{formatted_protein}\n")

    print(f"Generated proteins saved to {output_filename}")


if __name__ == "__main__":
    main_menu()

