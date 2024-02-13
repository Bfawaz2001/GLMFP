import pickle
import random

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
        # Start with a random amino acid
        current_aa = random.choice(list(model.keys()))
        protein.append(current_aa)

        while len(protein) < length:
            next_aa = random.choices(list(model[current_aa].keys()), weights=model[current_aa].values())[0]
            protein.append(next_aa)
            current_aa = next_aa

    elif model_type == '3mer':
        # Start with a random amino acid pair
        current_pair = random.choice(list(model.keys()))
        second_aa = random.choice(list(model[current_pair].keys()))
        protein.append(current_pair + second_aa)


        while len(protein) < length:
            try:
                next_aa = random.choices(list(model[current_pair][second_aa].keys()), weights=model[current_pair][second_aa].values())[0]
                protein.append(next_aa)
                current_pair = protein[-2][-1] + second_aa  # This line might also need adjustment based on your exact logic.
                second_aa = next_aa
            except KeyError:
                # Fallback: Choose the next amino acid randomly from all options
                all_aas = 'ACDEFGHIKLMNPQRSTVWY'  # Considered all 20 standard amino acids
                next_aa = random.choice(all_aas)
                protein.append(next_aa)
                # Adjust the logic here as needed to maintain continuity.

    elif model_type == '5mer':
        # Assume starting sequence of four amino acids
        start_seq = random.choice(list(model.keys()))
        protein = [start_seq]
        while len(''.join(protein)) < length:
            last_seq = ''.join(protein)[-4:]
            if last_seq in model and model[last_seq]:
                next_aa = random.choices(list(model[last_seq].keys()), weights=model[last_seq].values())[0]
                protein.append(next_aa)
            else:
                break  # Stop if no valid continuation is found
        protein = ''.join(protein)[:length]

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
    print("1. Bigram (2-mer)")
    print("2. Trigram (3-mer)")
    print("3. 5-mer")
    choice = input("Enter your choice (1, 2, or 3): ")

    #File path to the models
    PATH = "../data/models/"

    if choice == '1':
        model_type = '2mer'
        filename = PATH + '2mer_model.pkl'
    elif choice == '2':
        model_type = PATH + '3mer'
        filename = PATH + '3mer_model.pkl'
    elif choice == '3':
        model_type = '5mer'
        filename = PATH + '5mer_model.pkl'
    else:
        print("Invalid choice. Returning to main menu.")
        return

    model = load_model(filename)
    generate_proteins_interface(model, model_type)


def generate_proteins_interface(model, model_type):
    """
    Interface for generating proteins using the selected model.
    """
    base_filename = input("Enter the name for the generated proteins file (without extension): ")
    output_filename = f"{base_filename}.fasta"

    num_proteins = int(input("Enter the number of proteins to be created: "))
    min_length = int(input("Enter the minimum length of the proteins: "))
    max_length = int(input("Enter the maximum length of the proteins: "))

    with open(output_filename, 'w') as file:
        for i in range(num_proteins):
            protein = generate_protein(model, min_length, max_length, model_type)
            formatted_protein = '\n'.join(protein[j:j + 60] for j in range(0, len(protein), 60))
            file.write(f">Protein_{i + 1}\n{formatted_protein}\n")

    print(f"Generated proteins saved to {output_filename}")


if __name__ == "__main__":
    main_menu()
