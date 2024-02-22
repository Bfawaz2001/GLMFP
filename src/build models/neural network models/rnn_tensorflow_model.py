from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from Bio import SeqIO
import time
from tensorflow.keras.callbacks import ModelCheckpoint


def create_rnn_model(input_dim, output_dim, embedding_dim=64, rnn_units=128):
    """
    Creates a simple Recurrent Neural Network (RNN) model using TensorFlow.

    Args:
        input_dim (int): The size of the vocabulary (number of unique characters in the dataset).
        output_dim (int): The size of the output dimension, often the same as the input dimension for character prediction.
        embedding_dim (int): The size of the embedding vector for each character.
        rnn_units (int): The number of units in the RNN layer.

    Returns:
        A compiled TensorFlow Sequential model with an embedding layer, an RNN layer, and a dense output layer.
    """
    model = Sequential([
        Embedding(input_dim=input_dim, output_dim=embedding_dim),
        SimpleRNN(rnn_units, return_sequences=False),
        Dense(output_dim, activation='softmax')
    ])
    return model


def preprocess_sequences(sequences, label_encoder, max_length=None):
    """
    Preprocesses protein sequences for training by encoding characters to integers and padding.

    Args:
        sequences (list of str): The protein sequences to preprocess.
        label_encoder (LabelEncoder): A fitted sklearn LabelEncoder for converting characters to integers.
        max_length (int, optional): The maximum length for padding sequences. If None, uses the length of the longest sequence.

    Returns:
        numpy.ndarray: An array of preprocessed and padded sequences.
    """
    sequences_int = [label_encoder.transform(list(seq)) for seq in sequences]
    sequences_padded = pad_sequences(sequences_int, maxlen=max_length, padding='post')
    return sequences_padded


def load_and_preprocess_data(filepath, label_encoder, test_size=0.2, random_state=42):
    """
    Loads protein sequences from a FASTA file, preprocesses, and splits them into training and testing sets.

    Args:
        filepath (str): The path to the FASTA file containing protein sequences.
        label_encoder (LabelEncoder): A fitted sklearn LabelEncoder for converting characters to integers.
        test_size (float): The fraction of the data to reserve for the test set.
        random_state (int): A seed for the random number generator to ensure reproducible splits.

    Returns:
        tuple: Two tuples containing the training and testing datasets (X_train, X_test).
    """
    sequences = [str(record.seq) for record in SeqIO.parse(filepath, "fasta")]
    sequences_padded = preprocess_sequences(sequences, label_encoder)
    X_train, X_test = train_test_split(sequences_padded, test_size=test_size, random_state=random_state)
    return np.array(X_train), np.array(X_test)


def main(fasta_file):
    """
    Main function to execute the RNN model training and evaluation.

    Args:
        fasta_file (str): The path to the FASTA file containing protein sequences.
    """
    print("Starting the script...")

    # Define the model and training parameters.
    amino_acids = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    embedding_dim = 64
    rnn_units = 128
    batch_size = 32
    epochs = 10

    # Initialize and fit the Label Encoder with all possible amino acids.
    print("Initializing the Label Encoder...")
    label_encoder = LabelEncoder()
    label_encoder.fit(list(amino_acids))

    # Load and preprocess the protein sequence data.
    print("Loading and preprocessing data...")
    X_train, X_test = load_and_preprocess_data(fasta_file, label_encoder)

    # Create and compile the RNN model.
    print("Creating the RNN model...")
    model = create_rnn_model(input_dim=len(amino_acids), output_dim=len(amino_acids),
                             embedding_dim=embedding_dim, rnn_units=rnn_units)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Define a ModelCheckpoint callback to save the best model during training.
    checkpoint_callback = ModelCheckpoint(
        'protein_rnn_model_checkpoint.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)

    # Train the model with the training data and validation on the test set.
    print("Training the model...")
    start_time = time.time()
    model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size,
              validation_data=(X_test, X_test), callbacks=[checkpoint_callback])
    end_time = time.time()

    # Evaluate the model's performance on the test set.
    print("Evaluating the model...")
    loss, accuracy = model.evaluate(X_test, X_test)
    print(f'Test loss: {loss}, Test accuracy: {accuracy}')

    # Save the final model to disk.
    model.save('protein_rnn_model_final.h5')
    print("Model training and evaluation complete.")
    print("Time elapsed: ", end_time - start_time)


if __name__ == "__main__":
    fasta_file = '../../../data/training data/uniprot_sprot.fasta'  # Ensure this path points to your FASTA file.
    main(fasta_file)